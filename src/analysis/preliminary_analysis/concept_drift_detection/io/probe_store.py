"""probe（線形ヘッドの重み）の保存・読み込み（design.md §6.7）。

特徴量の寄与度分析（`concept_drift_cause`）が **距離×時期行列の分析と同じ probe** で Integrated Gradients を計算するため、
学習した probe をすべて保存する。

- **保存キーは `(project, 窓長, seed, 学習期間の末尾日)`** ＝ **版に依存しない**。
  probe は学習期間の末尾日と窓長だけで決まるので、隣り合う版で範囲が重なる部分を重複させない。
- **ファイルは `(project, 窓長, seed)` ごとに 1 つ**（`<project>/probes/w<窓長>/seed<k>.npz`）。
  1 probe 1 ファイルにすると数万個の小さいファイルができ、読み書きが遅くなるため。
- 1 probe は `nn.Linear(128, 1)` ＝ **129 個の数値**（重み 128 ＋ 切片 1）。
  全 6 project で約 33,000 個・30 ファイル・約 17MB。
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DATES = "train_end"     # npz 内のキー：学習期間の末尾日（ISO 文字列）
_WEIGHTS = "weights"     # npz 内のキー：(n, 129) ＝ 重み128 ＋ 切片1


def probe_dir(root: Path, project: str, window_days: int) -> Path:
    return Path(root) / project / "probes" / f"w{window_days}"


def probe_path(root: Path, project: str, window_days: int, seed_idx: int) -> Path:
    return probe_dir(root, project, window_days) / f"seed{seed_idx}.npz"


def head_to_vector(head) -> np.ndarray:
    """線形ヘッドを (129,) の 1 次元配列にする（重み128 ＋ 切片1）。"""
    linear = getattr(head, "net", None)
    if not hasattr(linear, "weight") or not hasattr(linear, "bias"):
        raise TypeError("probe の保存は線形ヘッド（HEAD_TYPE='linear'）のみ対応しています")
    w = linear.weight.detach().cpu().numpy().reshape(-1)
    b = linear.bias.detach().cpu().numpy().reshape(-1)
    return np.concatenate([w, b]).astype(np.float32)


class ProbeStore:
    """1 プロジェクト・1 窓長ぶんの probe を集めて、seed ごとの npz にまとめて書き出す。

    版をまたいで同じ末尾日の probe が再学習されても、**先に入ったものだけを残す**
    （probe は版に依存しないので中身は同じ）。
    """

    def __init__(self, root: Path, project: str, window_days: int):
        self.root = Path(root)
        self.project = project
        self.window_days = int(window_days)
        self._by_seed: dict[int, dict[str, np.ndarray]] = {}
        self._disabled = False   # 線形ヘッド以外だったら保存を諦める（本体は止めない）

    def add(self, seed_idx: int, train_end, head) -> None:
        """probe を 1 つ受け取る（既に同じ (seed, 末尾日) があれば何もしない）。"""
        if head is None or self._disabled:
            return
        key = train_end.isoformat() if isinstance(train_end, (date, datetime)) else str(train_end)
        bucket = self._by_seed.setdefault(int(seed_idx), {})
        if key not in bucket:
            try:
                bucket[key] = head_to_vector(head)
            except TypeError as e:
                # 長い実行の途中で落とさない。以降は保存を諦めて本体の計算を続ける。
                self._disabled = True
                logger.warning(f"[{self.project}] probe の保存を中止します（{e}）。"
                               f"特徴量の寄与度分析を動かすには HEAD_TYPE='linear' で回し直してください。")

    def sink(self):
        """`drift_matrix.build_matrices(probe_sink=...)` に渡す関数を返す。"""
        return self.add

    @property
    def n_probes(self) -> int:
        return sum(len(v) for v in self._by_seed.values())

    def save(self) -> list[Path]:
        """seed ごとに npz を書く。既存ファイルがあれば**併合**する（部分実行に対応）。"""
        out_dir = probe_dir(self.root, self.project, self.window_days)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for seed_idx, bucket in sorted(self._by_seed.items()):
            path = probe_path(self.root, self.project, self.window_days, seed_idx)
            merged = dict(_read_raw(path))       # 既存 → 新規で上書き
            merged.update(bucket)
            keys = sorted(merged)
            np.savez_compressed(path, **{_DATES: np.array(keys),
                                         _WEIGHTS: np.stack([merged[k] for k in keys])})
            paths.append(path)
        if paths:
            logger.info(f"[{self.project}] probe を保存: {len(paths)} ファイル / "
                        f"{self.n_probes} 個（窓長 {self.window_days} 日）→ {out_dir}")
        return paths


def _read_raw(path: Path) -> dict[str, np.ndarray]:
    """保存済み npz を {末尾日(ISO) -> (129,)} で読む。無ければ空。"""
    path = Path(path)
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as z:
        return {str(k): w for k, w in zip(z[_DATES], z[_WEIGHTS])}


def load_probes(root: Path, project: str, window_days: int, seed_idx: int) -> dict[date, np.ndarray]:
    """保存済み probe を {学習期間の末尾日 -> (129,)} で読む（特徴量の寄与度分析用）。"""
    return {date.fromisoformat(k): w for k, w in _read_raw(
        probe_path(root, project, window_days, seed_idx)).items()}
