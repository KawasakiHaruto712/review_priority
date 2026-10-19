"""特徴量の表を作る・書く・読む（design.md §2）。

特徴量やレコードの計算は pretrained_encoders のものをそのまま呼ぶ（値は今の計算と完全に同じ）。
レコードを Python のオブジェクトで溜めず、数値の配列のかたまりへ行ごとに詰めながら作る（§2.3）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.background_problem.common.data_loader import (
    DEFAULT_CHANGES_DIR_TEMPLATE, DEFAULT_RELEASES_SUMMARY_CSV, load_changes, load_release_dates)
from src.analysis.preliminary_analysis.feature_table import constants as C
from src.analysis.preliminary_analysis.feature_table.build import fingerprint
from src.analysis.preliminary_analysis.pretrained_encoders.dataset import record_builder
from src.analysis.preliminary_analysis.pretrained_encoders.features import feature_builder
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra import common, status, ticket, waiting
from src.analysis.preliminary_analysis.pretrained_encoders.features.extra.index import CoreIndex
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants as pre
from src.utils.bot_detection import BOT_ACCOUNTS_CSV, BotDetector, host_of

logger = logging.getLogger(__name__)

_EPOCH = datetime(1970, 1, 1)
_US = timedelta(microseconds=1)
_NAT = np.iinfo(np.int64).min          # datetime64 の NaT と同じ値
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _us(dt: datetime | None) -> int:
    """datetime（UTC。タイムゾーン付きなら UTC に直す）をエポックからのマイクロ秒に。None は NaT。"""
    if dt is None:
        return _NAT
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return (dt - _EPOCH) // _US


class _Chunks:
    """決まった行数の配列のかたまりに、1 行ずつ詰める（§2.3）。"""

    def __init__(self, n_feat: int, n_fill: int, rows: int = C.CHUNK_ROWS):
        self.n_feat, self.n_fill, self.rows = n_feat, n_fill, rows
        self.done: list[dict] = []
        self._new()

    def _new(self) -> None:
        r = self.rows
        self.cur = {"change_id": np.empty(r, np.int64), "t": np.empty(r, np.int64), "created": np.empty(r, np.int64),
                    "decision_time": np.empty(r, np.int64), "open_until": np.empty(r, np.int64),
                    "label": np.empty(r, np.float64), "features": np.empty((r, self.n_feat), np.float64),
                    "filled": np.zeros((r, self.n_fill), bool)}
        self.pos = 0

    def add(self, change_id, t, created, decision_time, open_until, label, feats, filled_idx) -> None:
        i, c = self.pos, self.cur
        c["change_id"][i] = int(change_id)
        c["t"][i] = _us(t)
        c["created"][i] = _us(created)
        c["decision_time"][i] = _us(decision_time)
        c["open_until"][i] = _us(open_until)
        c["label"][i] = label
        c["features"][i] = feats
        if filled_idx:
            c["filled"][i, filled_idx] = True
        self.pos += 1
        if self.pos == self.rows:
            self.done.append(c)
            self._new()

    def finish(self) -> dict[str, np.ndarray]:
        """かたまりをつなげ、(t, 番号) の昇順に並べる（§2.1）。"""
        parts = self.done + [{k: v[:self.pos] for k, v in self.cur.items()}]
        out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        order = np.lexsort((out["change_id"], out["t"]))
        out = {k: v[order] for k, v in out.items()}
        for k in ("t", "created", "decision_time", "open_until"):
            out[k] = out[k].view("datetime64[us]")
        return out


def build_arrays(changes: list[dict], project: str, pool_start: datetime, cycle_end: datetime,
                 detector: BotDetector, all_prs: pd.DataFrame, rel_df: pd.DataFrame,
                 main_branches: set[str]) -> tuple[dict[str, np.ndarray], dict]:
    """事前学習と同じレコード（record_builder.iter_records）に 67 個の特徴量と代わりの値の印を付けた表の配列と、集計を返す。"""
    facts_by_idx = [common.build_change_facts(ch, project, detector) for ch in changes]
    facts = [f for f in facts_by_idx if f is not None]
    sidx = status.StatusIndex(facts, project, CoreIndex(facts))
    fill_pos = {n: k for k, n in enumerate(C.FILLABLE_NAMES)}
    feat_pos = {n: k for k, n in enumerate(C.ALL_FEATURE_NAMES)}
    chunks = _Chunks(len(C.ALL_FEATURE_NAMES), len(C.FILLABLE_NAMES))
    stats = {"no_revision_records": 0}
    for idx, rec in record_builder.iter_records(changes, project, pool_start, cycle_end, detector, all_prs, rel_df):
        f = facts_by_idx[idx]
        if f is None:          # 作成時刻のない Change はレコードにならないので起きない
            raise RuntimeError(f"[{project}] レコードの Change {rec.change_id} の ChangeFacts がありません")
        if f.latest_revision(rec.t) is None:
            stats["no_revision_records"] += 1          # 起きないはず（§12.4）。起きたら記録に残す
        filled = set(rec.filled)
        tv = ticket.ticket_features(f, rec.t, project, main_branches)
        sv = sidx.features(f, rec.t, filled)
        wv = waiting.waiting_features(f, rec.t, project, filled)
        row = list(rec.features) + [tv[k] for k in ticket.TICKET_FEATURES] \
            + [sv[k] for k in status.STATUS_FEATURES] + [wv[k] for k in waiting.WAITING_FEATURES]
        chunks.add(rec.change_id, rec.t, rec.created, rec.decision_time, rec.open_until,
                   rec.labels[pre.TARGET], row, [fill_pos[n] for n in filled if n in fill_pos])
    arr = chunks.finish()
    feats = arr["features"]
    stats["rows"] = int(len(feats))
    stats["positives"] = int(np.sum(arr["label"] == 1.0))
    stats["filled_counts"] = {n: int(arr["filled"][:, k].sum()) for k, n in enumerate(C.FILLABLE_NAMES)}
    # 代わりの値も出せなかった −1（起きないはず）。符号のある特徴量は除く
    stats["minus_one_counts"] = {n: int(np.sum(feats[:, feat_pos[n]] == -1.0))
                                 for n in C.ALL_FEATURE_NAMES
                                 if n not in C.SIGNED_FEATURES and np.any(feats[:, feat_pos[n]] == -1.0)}
    if stats["no_revision_records"] or stats["minus_one_counts"]:
        logger.warning(f"[{project}] 版のないレコード {stats['no_revision_records']} 件、−1 の残り {stats['minus_one_counts']}")
    return arr, stats


def save_table(arr: dict[str, np.ndarray], out_dir: Path, period: str = C.PERIOD_PRETRAIN) -> Path:
    """npz（圧縮なし。§2.4）で書く。名前の並びも一緒に持つ。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{period}.npz"
    np.savez(path, **arr,
             feature_names=np.array(C.ALL_FEATURE_NAMES, dtype=str),
             filled_names=np.array(C.FILLABLE_NAMES, dtype=str))
    return path


def load_table(project: str, period: str = C.PERIOD_PRETRAIN, features: list[str] | None = None,
               root: Path = C.TABLE_ROOT) -> dict[str, np.ndarray]:
    """表を読む（§2.6）。features で列を選べる（並びは指定どおり）。"""
    with np.load(Path(root) / project / f"{period}.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    names = [str(n) for n in out["feature_names"]]
    if features is not None:
        pos = {n: i for i, n in enumerate(names)}
        missing = [n for n in features if n not in pos]
        if missing:
            raise KeyError(f"表にない特徴量: {missing}")
        out["features"] = out["features"][:, [pos[n] for n in features]]
        out["feature_names"] = np.array(features, dtype=str)
    return out


# ── 作った条件の記録（§2.5） ─────────────────────────────────
def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _changes_dir_summary(project: str) -> dict:
    d = Path(str(DEFAULT_CHANGES_DIR_TEMPLATE).format(project=project))
    files = sorted(d.glob("*.json")) if d.exists() else []
    stats = [p.stat() for p in files]
    return {"path": str(d), "files": len(files), "total_bytes": int(sum(s.st_size for s in stats)),
            "latest_mtime": (datetime.fromtimestamp(max(s.st_mtime for s in stats), timezone.utc).isoformat()
                             if stats else None)}


def build_meta(project: str, period: str, rng: dict, pool_start: datetime, cycle_end: datetime,
               observation_end: datetime, stats: dict, elapsed_sec: float, fp: dict | None) -> dict:
    """作った条件の記録（§2.5）。fp は指紋（fingerprint.compute の値）。"""
    porcelain = _git("status", "--porcelain")
    return {
        "project": project, "period": period, "created_at": datetime.now(timezone.utc).isoformat(),
        "code": {"commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(porcelain) if porcelain is not None else None},
        "constants": {"MEASUREMENT_STEP_DAYS": pre.MEASUREMENT_STEP_DAYS, "LOOKBACK_DAYS": pre.LOOKBACK_DAYS,
                      "REVIEW_HORIZON_DAYS": pre.REVIEW_HORIZON_DAYS, "TARGET": pre.TARGET, "range": rng,
                      "pool_start": pool_start.isoformat(), "cycle_end": cycle_end.isoformat(),
                      "observation_end": observation_end.isoformat()},
        "inputs": {"changes": _changes_dir_summary(project),
                   "major_releases_summary_sha256": _sha256(DEFAULT_RELEASES_SUMMARY_CSV),
                   "bot_accounts_sha256": _sha256(BOT_ACCOUNTS_CSV),
                   "main_branches_sha256": _sha256(ticket.MAIN_BRANCHES_CSV)},
        "feature_names": C.ALL_FEATURE_NAMES, "filled_names": C.FILLABLE_NAMES,
        "result": stats,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "elapsed_sec": round(elapsed_sec, 1)},
        "fingerprint": fp,
    }


def table_paths(project: str, period: str, root: Path = C.TABLE_ROOT) -> tuple[Path, Path]:
    out_dir = Path(root) / project
    return out_dir / f"{period}.npz", out_dir / f"{period}_meta.json"


class TableError(RuntimeError):
    """表が使えない（指紋がない・指紋が違う・すでにある）。上書きしないために止める（§2.6）。"""


def build_project(project: str, period: str = C.PERIOD_PRETRAIN, root: Path = C.TABLE_ROOT,
                  overwrite: bool = False) -> Path:
    """1 件のリポジトリの、その期間の表を作って書く（§2.1）。すでにあれば、overwrite でなければ止める（上書き＝消すことになるため）。"""
    path, meta_path = table_paths(project, period, root)
    if path.exists() and not overwrite:
        raise TableError(f"[{project}] {period} の表はすでにあります：{path}（作り直すなら --overwrite）")
    t0 = time.perf_counter()
    rel_df = load_release_dates()
    fp = fingerprint.compute(project, period, rel_df)           # 作る前の時点のコード・データで指紋を出す
    rng = fp["parts"]["constants"]["range"]
    changes = load_changes(project)
    if not changes:
        raise RuntimeError(f"[{project}] Change が読めませんでした")
    detector = BotDetector(host_of(project))
    all_prs = feature_builder.build_all_prs_df(changes, detector)
    observation_end = pd.Timestamp(all_prs["updated"].max()).to_pydatetime()   # record_builder の観測の終わりと同じ
    if period == C.PERIOD_PRETRAIN:
        pool_start = all_prs["created"].min().to_pydatetime()                  # データの始まり
    else:
        pool_start = datetime.fromisoformat(rng["start"])
    cycle_end = datetime.fromisoformat(rng["end"])
    logger.info(f"[{project}] {period} の表を作る：{pool_start} 〜 {cycle_end}")
    arr, stats = build_arrays(changes, project, pool_start, cycle_end, detector, all_prs, rel_df,
                              ticket.load_main_branches(project))
    path = save_table(arr, path.parent, period)
    meta = build_meta(project, period, rng, pool_start, cycle_end, observation_end, stats, time.perf_counter() - t0, fp)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    logger.info(f"[{project}] 書いた：{path}（{stats['rows']} 行、{meta['environment']['elapsed_sec']} 秒）")
    return path


def check_table(project: str, period: str = C.PERIOD_PRETRAIN, root: Path = C.TABLE_ROOT,
                rel_df: pd.DataFrame | None = None) -> None:
    """表の指紋が今のコード・データと同じかを確かめる。違えば TableError（§2.6）。"""
    _path, meta_path = table_paths(project, period, root)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    old = meta.get("fingerprint")
    if not old:
        raise TableError(f"[{project}] {period} の表に指紋がありません。値に効くファイルとデータを表を作った後に変えていなければ、"
                         f"python3 -m src.analysis.preliminary_analysis.feature_table.build --add-fingerprint {project} で書き足してください")
    cur = fingerprint.compute(project, period, load_release_dates() if rel_df is None else rel_df)
    if cur["sha256"] != old["sha256"]:
        diffs = fingerprint.differences(old.get("parts", {}), cur["parts"])
        raise TableError(f"[{project}] {period} の表は今のコード・データと違う条件で作られています（違い：{', '.join(diffs)}）。"
                         f"作り直すなら：python3 -m src.analysis.preliminary_analysis.feature_table.build {project} "
                         f"--period {period} --overwrite（古い表は上書きされます）")


def ensure_table(project: str, period: str = C.PERIOD_PRETRAIN, features: list[str] | None = None,
                 root: Path = C.TABLE_ROOT, rel_df: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """表を使う側の入口（§2.6）：なければ作る、指紋がない・違えば止める（上書きしない）、そろっていれば読む。"""
    path, _meta = table_paths(project, period, root)
    if not path.exists():
        logger.warning(f"[{project}] {period} の表がないので、今から作ります（数時間かかることがあります）")
        build_project(project, period, root)
    check_table(project, period, root, rel_df)
    return load_table(project, period, features, root)


def add_fingerprint(project: str, period: str = C.PERIOD_PRETRAIN, root: Path = C.TABLE_ROOT) -> dict:
    """指紋のない表（2026-10-18 に指紋を入れる前に作った表）に、今のコード・データの指紋を書き足す（§2.5）。

    表を作った後に値に効くファイルもデータも変えていないことが前提（本人に確かめてから使う）。指紋がすでにあれば止める。
    """
    _path, meta_path = table_paths(project, period, root)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint"):
        raise TableError(f"[{project}] {period} の表にはすでに指紋があります")
    meta["fingerprint"] = fingerprint.compute(project, period, load_release_dates())
    meta["fingerprint_added_at"] = datetime.now(timezone.utc).isoformat()
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta["fingerprint"]
