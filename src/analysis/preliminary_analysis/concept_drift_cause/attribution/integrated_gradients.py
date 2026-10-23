"""Integrated Gradients（design.md §2, §3）。

1 評価日・1 モデルについて、**その日の全 Change** の 15 次元 IG を計算し、
**正解ラベルで正例群／負例群に分けて平均**した 15 次元プロファイルを返す。

計算の形（§2.3）:
    基準点 ＝ その評価日の集合の平均ベクトル
    経路   ＝ **対象 Change だけ**を基準点から実際の値まで動かす（他の Change は実際の値で固定）
    出力   ＝ **ロジット**（確率だとシグモイドの飽和で潰れる。§2.2）
    IG_i   ＝ (実際の値 − 基準点)_i × 経路上の勾配の平均

他の Change を固定するのは、エンコーダが自己注意を使うため。集合ごと動かすと IG が
「自分の特徴量の寄与」と「周囲が変わった影響」の混合物になり、15 特徴量に配分する意味が消える。
"""
from __future__ import annotations

import numpy as np
import torch

from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.model.set_transformer import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


def _alphas(steps: int, device) -> torch.Tensor:
    """積分の刻み（中点則）。(steps, 1)。両端に寄らないぶん右端則より誤差が小さい。"""
    return ((torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps).view(steps, 1)


def item_attributions(encoder, head, feats_std: np.ndarray, steps: int, device,
                      items=None, ctx_block: np.ndarray | None = None, step_chunk: int | None = None) -> np.ndarray:
    """その日の Change ごとの IG を (len(items), 15) で返す（items 省略時は全件）。

    feats_std : (L, 15) 標準化済みの特徴（`st.standardize_feats`）
    ctx_block : (L, L) の文脈のマスク（True=見えない。`st.context_block_mask(集合)`。pretrained_encoders/design.md §12.4）。
                None なら全員が互いに見える。距離×時期行列の分析と同じ文脈でロジットを計算するために渡す。
    step_chunk: 一度に計算する段階の数（design.md §2.4。2026-10-20）。段階を分けて勾配を足し、最後に steps で割るので値は変わらない。
                None なら全段階をまとめて 1 回（旧版と同じ）。かたまりごとに GPU のキャッシュを空にする。
    """
    n_items = feats_std.shape[0]
    idx = list(range(n_items)) if items is None else list(items)
    out = np.zeros((len(idx), N_FEATURES), dtype=np.float64)
    if n_items == 0 or not idx:
        return out

    x = torch.from_numpy(feats_std).to(device)
    base = x.mean(dim=0)                                   # その日の集合の平均（§2.3）
    alphas = _alphas(steps, device)
    chunk = steps if step_chunk is None else max(1, min(int(step_chunk), steps))
    ctx_one = None if ctx_block is None else torch.from_numpy(np.asarray(ctx_block, dtype=bool)).to(device).unsqueeze(0)

    for i, j in enumerate(idx):
        delta = x[j] - base                                 # (15,)
        grad_sum = torch.zeros_like(delta)
        # 段階ごとに「対象 j だけ」基準点→実際の値へ動かした集合を作る。chunk 個ずつ計算して勾配を足す（§2.4）
        for a in range(0, steps, chunk):
            al = alphas[a:a + chunk]
            m = al.shape[0]
            valid = torch.ones(m, n_items, dtype=torch.bool, device=device)
            ctx = None if ctx_one is None else ctx_one.repeat(m, 1, 1)
            moved = (base.unsqueeze(0) + al * delta.unsqueeze(0)).requires_grad_(True)
            batch = x.unsqueeze(0).repeat(m, 1, 1)
            batch = torch.cat([batch[:, :j, :], moved.unsqueeze(1), batch[:, j + 1:, :]], dim=1)
            logits = st.item_logits(encoder, head, batch, valid, ctx)[:, j]   # 対象 j のロジットのみ
            logits.sum().backward()
            grad_sum += moved.grad.detach().sum(dim=0)
            del valid, ctx, moved, batch, logits
            if chunk < steps:
                st.release_device_cache(device)
        out[i] = ((grad_sum / steps) * delta).cpu().numpy()   # 経路上の勾配の平均 × (実際の値 − 基準点)
    return out


def day_profiles(encoder, head, feats_std: np.ndarray, labels, steps: int, device,
                 ctx_block: np.ndarray | None = None, step_chunk: int | None = None, items=None) -> dict:
    """1 評価日・1 モデルの {群 -> 15 次元プロファイル} と件数を返す（§3.1）。

    群分けは**正解ラベル**（モデルに依存しないので、別のモデル同士でも同じ Change 群を比べられる）。
    群の中は単純平均。
    items（2026-10-23。design.md §11.7）：IG を計算する Change の位置（標本）。省略時は全件。
    文脈（モデルへの入力の集合）は items によらず全件のまま。件数は計算した Change の数。
    """
    y = np.asarray(labels, dtype=float)
    if items is not None:
        y = y[np.asarray(list(items), dtype=int)]
    attrs = item_attributions(encoder, head, feats_std, steps, device, items=items,
                              ctx_block=ctx_block, step_chunk=step_chunk)
    pos = attrs[y == 1]
    neg = attrs[y == 0]
    return {
        "pos": pos.mean(axis=0) if len(pos) else np.full(N_FEATURES, np.nan),
        "neg": neg.mean(axis=0) if len(neg) else np.full(N_FEATURES, np.nan),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
