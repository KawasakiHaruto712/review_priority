"""リポジトリごとに別々のプロセスで並べて回す（pretrained_encoders/design.md §14.7。2026-10-21）。

- CUDA が使える機械（サーバ）なら並べる。MPS・CPU（Mac など）なら 1 つずつ（Mac はメモリ 16 GB を CPU と GPU で分け合うため）。
- GPU を使う分析：--gpu-share（並べたプロセス全体で使ってよい GPU のメモリの割合。既定 0.25）から、1 つあたり約 1.6 GB を確保できる数にする
  （リポジトリの数と CPU の上限を超えない）。子のプロセスの上限 ＝ 割合 ÷ 並べる数（環境変数 RP_CUDA_MEMORY_FRACTION で渡す）。
- GPU を使わない分析：CPU の上限だけで決める（サーバなら最大 CPU_ONLY_MAX）。
- ログはリポジトリごとに logs/<名前>_<リポジトリ>_<日時>.log。最後にまとめを logs/<名前>_summary_<日時>.log に出す。失敗しても、ほかは止めない。
各分析で独立させるため、同じものを concept_drift_detection・data_drift にも複製して置く（整理は論文の後）。
"""
from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# 1 つのプロセスに確保したい GPU のメモリ。~~1.6~~ 2026-10-21 改めた：2.5（feature_space_window/design.md §9.4）。
# RQ2 は事前学習の区間の日も埋め込むので、qtbase（集合が最大 2,073 件）の埋め込みの山が評価期間だけのとき（1.07 GB）の約 2 倍になり、
# 1.64 GB の上限で CUDA out of memory になった。RQ1 の分析（concept_drift_detection・data_drift の複製）は 1.6 のまま。
PER_PROCESS_GB = 2.5
CPU_ONLY_MAX = 6           # GPU を使わない分析を並べる数の上限
DEFAULT_GPU_SHARE = 0.25   # 全体の 1/4（研究室の決まり）


def cpu_limit() -> int:
    """使える CPU の数（Docker の上限 cgroup があればそれ、なければ os.cpu_count）。"""
    for path in ("/sys/fs/cgroup/cpu.max",):
        try:
            quota, period = Path(path).read_text().split()[:2]
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
        except (OSError, ValueError):
            pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _cuda_total_gb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    return None


def decide(n_items: int, *, use_gpu: bool, gpu_share: float = DEFAULT_GPU_SHARE, workers: int | None = None,
           total_gb: float | None = -1.0, cpus: int | None = None) -> tuple[int, float | None]:
    """(並べる数, 子のプロセスの GPU のメモリの割合 or None)。total_gb・cpus はテスト用に渡せる（-1.0 なら調べる）。"""
    total = _cuda_total_gb() if total_gb == -1.0 else total_gb
    cpus = cpu_limit() if cpus is None else cpus
    if workers:
        w = max(1, min(int(workers), n_items))
    elif total is None:
        w = 1                                         # CUDA がない（Mac など）は 1 つずつ
    elif use_gpu:
        w = max(1, min(n_items, cpus, math.floor(gpu_share * total / PER_PROCESS_GB)))
    else:
        w = max(1, min(n_items, cpus, CPU_ONLY_MAX))
    frac = (gpu_share / w) if (use_gpu and total is not None) else None
    return w, frac


def run_children(module: str, projects: list[str], child_args: list[str], workers: int, fraction: float | None,
                 name: str, log_dir: Path = Path("logs"), poll: float = 5.0) -> list[str]:
    """`python -m module --project p <child_args>` をリポジトリごとに並べて回し、失敗したリポジトリを返す。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ)
    if fraction is not None:
        env["RP_CUDA_MEMORY_FRACTION"] = f"{fraction:.4f}"
    queue, running, results = list(projects), {}, {}
    logger.info(f"{name}：{len(projects)} 件を {workers} つ並べて回します"
                + (f"（1 つあたり GPU のメモリの {fraction:.1%}）" if fraction is not None else ""))
    while queue or running:
        while queue and len(running) < workers:
            p = queue.pop(0)
            log = log_dir / f"{name}_{p}_{stamp}.log"
            fh = open(log, "w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, "-m", module, "--project", p, "--workers", "1", *child_args],
                                    stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[p] = (proc, fh, log)
            logger.info(f"開始：{p}（ログ {log}）")
        for p, (proc, fh, log) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                results[p] = (proc.returncode, log)
                del running[p]
                logger.info(f"{'終了' if proc.returncode == 0 else '失敗'}：{p}（終了コード {proc.returncode}）")
        if running or queue:
            time.sleep(poll)
    failed = [p for p, (rc, _l) in results.items() if rc != 0]
    summary = log_dir / f"{name}_summary_{stamp}.log"
    with open(summary, "w", encoding="utf-8") as f:
        for p in projects:
            rc, log = results.get(p, (None, None))
            f.write(f"{p}\t{'OK' if rc == 0 else 'FAILED'}\t終了コード {rc}\t{log}\n")
        f.write(f"workers={workers}\tgpu_fraction_per_process={fraction}\n")
    logger.info(f"まとめ：{summary}（失敗 {failed or 'なし'}）")
    return failed
