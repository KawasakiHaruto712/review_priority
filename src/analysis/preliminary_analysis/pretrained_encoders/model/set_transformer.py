"""集合Transformer（自己注意）＋転移学習の本体（design.md §3, §4）。

構成：
- Scaler       : 特徴の標準化（事前学習データで fit → 全体に適用）
- SetEncoder   : 位置エンコなしの自己注意エンコーダ（置換不変・可変長）。各 Change に文脈込み表現を出す。
- Head         : 予測ヘッド。linear probe（線形1層）／small MLP を選択。
- pretrain     : 教師あり事前学習（エンコーダ＋仮ヘッドで「Δ以内レビュー」を予測）→ エンコーダ凍結。
- embed_sets   : 凍結エンコーダで各集合の per-item 埋め込みを 1 回計算（キャッシュ）。
- train_probe  : キャッシュ済み埋め込みの上でヘッドだけ学習（linear probing）。下流が使う。
- predict      : ヘッドで per-Change 予測（正例確率）を返す。下流が使う。

「仮ヘッド（汎用ヘッド）」は評価に使うため破棄せず返す（保存対象）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.analysis.preliminary_analysis.pretrained_encoders.dataset.set_builder import TSet
from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_builder import FEATURE_NAMES
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants

FEATURE_DIM = len(FEATURE_NAMES)  # 15


# ── デバイス（MPS があれば MPS、無ければ CPU） ─────────────────────
def resolve_device() -> torch.device:
    want = constants.DEVICE
    if want == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(want)


# ── 特徴の標準化 ────────────────────────────────────
class Scaler:
    """特徴を平均0・分散1に標準化する（事前学習データで統計量を fit）。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.where(std < 1e-8, 1.0, std)  # 分散0の特徴で割らないよう保護

    @classmethod
    def fit(cls, sets: list[TSet]) -> "Scaler":
        feats = np.array([f for s in sets for f in s.feats], dtype=np.float64)
        if feats.size == 0:
            return cls(np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM))
        return cls(feats.mean(axis=0), feats.std(axis=0))

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return (feats - self.mean) / self.std


# ── バッチ化（集合を padding + mask してテンソルに） ──────────────────
def _pad_batch(sets: list[TSet], scaler: Scaler, device: torch.device):
    """集合のリストを (feats, labels, valid_mask) テンソルにパディングする。

    feats: (B, Lmax, 15) 標準化済み / labels: (B, Lmax) / valid_mask: (B, Lmax) True=有効, False=パディング
    """
    b = len(sets)
    lmax = max(len(s) for s in sets)
    feats = np.zeros((b, lmax, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((b, lmax), dtype=np.float32)
    valid = np.zeros((b, lmax), dtype=bool)
    for i, s in enumerate(sets):
        n = len(s)
        feats[i, :n] = scaler.transform(np.array(s.feats, dtype=np.float64))
        labels[i, :n] = np.array(s.labels, dtype=np.float32)
        valid[i, :n] = True
    return (torch.from_numpy(feats).to(device),
            torch.from_numpy(labels).to(device),
            torch.from_numpy(valid).to(device))


# ── エンコーダ（自己注意・位置エンコなし＝集合として扱う。design.md §3） ──
class SetEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(FEATURE_DIM, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="relu")
        # enable_nested_tensor=False: パディングマスク使用時の nested-tensor 高速化パスを無効化。
        # 当該 op が MPS 未対応で eval 時に落ちるため（正攻法の回避）。精度・挙動は変わらない。
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        # 注：位置エンコーディングは足さない → 置換不変（集合として扱う）

    def forward(self, feats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """feats:(B,L,15) → per-item 埋め込み (B,L,d_model)。valid_mask:True=有効。"""
        x = self.input_proj(feats)
        # TransformerEncoder の src_key_padding_mask は True=無視（パディング）なので反転して渡す
        out = self.encoder(x, src_key_padding_mask=~valid_mask)
        return out


# ── ヘッド（linear probe / small MLP） ─────────────────────
class Head(nn.Module):
    def __init__(self, d_model: int, head_type: str, hidden: int):
        super().__init__()
        if head_type == "mlp":
            self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:  # "linear"（linear probe, 既定）
            self.net = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb:(*, d_model) → ロジット (*,)。"""
        return self.net(emb).squeeze(-1)


def _make_head() -> Head:
    return Head(constants.D_MODEL, constants.HEAD_TYPE, constants.HEAD_HIDDEN)


def _iter_batches(sets: list[TSet], batch_sets: int):
    for i in range(0, len(sets), batch_sets):
        yield sets[i:i + batch_sets]


# ── 教師あり事前学習（design.md §4） ──────────────────────────
def pretrain(pretrain_sets: list[TSet], scaler: Scaler, seed: int, device: torch.device):
    """エンコーダ＋仮ヘッドを「Δ以内レビュー(0/1)」予測で学習し、（凍結）エンコーダと仮ヘッドを返す。

    仮ヘッド（＝汎用ヘッド）は評価・保存に使うため破棄せず返す。
    """
    torch.manual_seed(seed)
    encoder = SetEncoder(constants.D_MODEL, constants.N_LAYERS, constants.N_HEADS,
                         constants.FFN_DIM, constants.DROPOUT).to(device)
    head = _make_head().to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=constants.PRETRAIN_LR)
    loss_fn = nn.BCEWithLogitsLoss()

    encoder.train(); head.train()
    order = list(range(len(pretrain_sets)))
    rng = np.random.default_rng(seed)
    for _ in range(constants.PRETRAIN_EPOCHS):
        rng.shuffle(order)
        shuffled = [pretrain_sets[i] for i in order]
        for batch in _iter_batches(shuffled, constants.PRETRAIN_BATCH_SETS):
            feats, labels, valid = _pad_batch(batch, scaler, device)
            emb = encoder(feats, valid)
            logits = head(emb)                    # (B, L)
            m = valid.reshape(-1)
            loss = loss_fn(logits.reshape(-1)[m], labels.reshape(-1)[m])
            opt.zero_grad(); loss.backward(); opt.step()

    # エンコーダを凍結（以降は推論のみ）
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    head.eval()
    return encoder, head


# ── 埋め込みキャッシュ（下流が使う） ─────────────────────────────
@torch.no_grad()
def embed_sets(encoder: SetEncoder, scaler: Scaler, sets: list[TSet], device: torch.device) -> list[np.ndarray]:
    """凍結エンコーダで各集合の per-item 埋め込みを計算（キャッシュ用）。集合ごとに (L, d_model) を返す。"""
    out: list[np.ndarray] = []
    for batch in _iter_batches(sets, constants.PRETRAIN_BATCH_SETS):
        feats, _labels, valid = _pad_batch(batch, scaler, device)
        emb = encoder(feats, valid).cpu().numpy()  # (B, Lmax, d_model)
        for i, s in enumerate(batch):
            out.append(emb[i, :len(s)])             # パディング除去
    return out


# ── linear probing（キャッシュ埋め込みの上でヘッドだけ学習。下流が使う） ──────
def _flatten(embeds: list[np.ndarray], sets: list[TSet]):
    """集合ごとの埋め込み/ラベルを (N, d_model), (N,) に平坦化（per-item 学習用）。"""
    X = np.concatenate(embeds, axis=0) if embeds else np.zeros((0, constants.D_MODEL), np.float32)
    y = np.array([lab for s in sets for lab in s.labels], dtype=np.float32)
    return X, y


def train_probe(train_embeds: list[np.ndarray], train_sets: list[TSet], seed: int, device: torch.device) -> Head:
    """凍結表現の上で per-item のヘッドを学習（linear probe）。片クラスのみなら None を返す。"""
    X, y = _flatten(train_embeds, train_sets)
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        return None
    torch.manual_seed(seed)
    head = _make_head().to(device)
    opt = torch.optim.Adam(head.parameters(), lr=constants.PROBE_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    head.train()
    for _ in range(constants.PROBE_EPOCHS):
        logits = head(xt)
        loss = loss_fn(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict(head: Head, eval_embeds: list[np.ndarray], eval_sets: list[TSet], device: torch.device):
    """ヘッドで per-Change の正例確率を予測し、(y_true, y_pred, change_id, t) の行を返す。"""
    rows = []
    for emb, s in zip(eval_embeds, eval_sets):
        if len(s) == 0:
            continue
        xt = torch.from_numpy(emb).to(device)
        prob = torch.sigmoid(head(xt)).cpu().numpy()
        for k in range(len(s)):
            rows.append((float(s.labels[k]), float(prob[k]), s.ids[k], s.t))
    return rows


# ── 勾配を通す経路（特徴量の寄与度分析 / concept_drift_cause の Integrated Gradients 用） ──────────
# 上の embed_sets / predict は @torch.no_grad() ＋ numpy キャッシュのため、入力まで勾配が
# たどれない。IG は「入力を動かしたとき出力がどう動くか」を見るので、その経路を別に用意する。
# **既存の関数は変更しない**（lookback_window / concept_drift_detection に影響させないため）。

def standardize_feats(scaler: Scaler, feats) -> np.ndarray:
    """集合の特徴を標準化した (L, 15) の float32 配列にする（パディングしない）。

    `_pad_batch` と同じ変換だが、IG は補間したテンソルを自分で組み立てるので単体で使えるものが要る。
    なお**標準化は IG の値に影響しない**（勾配側と差分側で打ち消し合う）ので、
    標準化後の空間で補間してよい（concept_drift_cause/design.md §2.3）。
    """
    return scaler.transform(np.asarray(feats, dtype=np.float64)).astype(np.float32)


def head_from_vector(vec: np.ndarray, device: torch.device = None) -> Head:
    """保存済みの (d_model+1,) ＝ 重み d_model ＋ 切片 1 から線形ヘッドを復元する。

    距離×時期行列の分析が保存した probe（concept_drift_detection/io/probe_store.py）を読み戻すために使う。
    """
    head = Head(constants.D_MODEL, "linear", constants.HEAD_HIDDEN)
    with torch.no_grad():
        head.net.weight.copy_(torch.as_tensor(vec[:-1], dtype=torch.float32).view(1, -1))
        head.net.bias.copy_(torch.as_tensor(vec[-1:], dtype=torch.float32))
    head.eval()
    return head.to(device) if device is not None else head


def freeze_for_gradients(*modules) -> None:
    """パラメータ側の勾配を切る（入力側だけ勾配を流すため）。

    IG で欲しいのは入力に対する勾配だけ。重みの `.grad` が溜まると無駄にメモリを食うので落とす。
    """
    for mod in modules:
        if mod is None:
            continue
        for prm in mod.parameters():
            prm.requires_grad_(False)


def item_logits(encoder: SetEncoder, head: Head, feats: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
    """標準化済み特徴 (B, L, 15) から per-item の**ロジット** (B, L) を返す（勾配を通す）。

    確率ではなくロジットを返すのは、シグモイドの飽和で IG が潰れるのを避けるためと、
    線形ヘッドの分解（IG ＝ 翻訳表 × 重み）を保つため（concept_drift_cause/design.md §2.2, §2.5）。
    """
    return head(encoder(feats, valid))
