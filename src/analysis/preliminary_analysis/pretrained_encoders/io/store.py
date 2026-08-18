"""事前学習済みエンコーダ＋汎用ヘッド＋Scaler＋config の保存/読込（design.md §6）。

保存レイアウト（下流との約束事）:
  <OUTPUT_ROOT>/<project>/cutoff_<cutoff>/seed<k>/
    encoder.pt / general_head.pt / scaler.json / config.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_builder import FEATURE_NAMES
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants


def _seed_dir(project: str, cutoff: str, seed_idx: int) -> Path:
    return constants.OUTPUT_ROOT / project / f"cutoff_{cutoff}" / f"seed{seed_idx}"


def _base_config(project: str, cutoff: str, seed_idx: int) -> dict:
    return {
        "project": project, "cutoff": cutoff, "seed": seed_idx,
        "feature_names": FEATURE_NAMES, "feature_dim": len(FEATURE_NAMES),
        "d_model": constants.D_MODEL, "n_layers": constants.N_LAYERS, "n_heads": constants.N_HEADS,
        "ffn_dim": constants.FFN_DIM, "dropout": constants.DROPOUT, "max_set_size": constants.MAX_SET_SIZE,
        "head_type": constants.HEAD_TYPE, "head_hidden": constants.HEAD_HIDDEN,
        "pretrain_method": constants.PRETRAIN_METHOD, "pretrain_epochs": constants.PRETRAIN_EPOCHS,
        "pretrain_lr": constants.PRETRAIN_LR, "pretrain_batch_sets": constants.PRETRAIN_BATCH_SETS,
        "review_horizon_days": constants.REVIEW_HORIZON_DAYS,
        "measurement_step_days": constants.MEASUREMENT_STEP_DAYS, "lookback_days": constants.LOOKBACK_DAYS,
    }


def save_pretrained(project: str, cutoff: str, seed_idx: int, encoder, general_head, scaler,
                    extra_config: dict | None = None) -> Path:
    """1 seed 分（encoder＋汎用ヘッド＋scaler＋config）を保存し、保存先を返す。"""
    d = _seed_dir(project, cutoff, seed_idx)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), d / "encoder.pt")
    torch.save(general_head.state_dict(), d / "general_head.pt")
    with open(d / "scaler.json", "w", encoding="utf-8") as f:
        json.dump({"mean": np.asarray(scaler.mean).tolist(),
                   "std": np.asarray(scaler.std).tolist()}, f, ensure_ascii=False, indent=2)
    config = _base_config(project, cutoff, seed_idx)
    config.update(extra_config or {})
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return d


def list_seeds(project: str, cutoff: str) -> list[int]:
    """保存済み seed（0 始まりの反復 index）を昇順で返す。"""
    base = constants.OUTPUT_ROOT / project / f"cutoff_{cutoff}"
    if not base.exists():
        return []
    return sorted(int(p.name[4:]) for p in base.glob("seed*") if p.name[4:].isdigit())


def load_pretrained(project: str, cutoff: str, seed_idx: int, device=None):
    """保存済み (encoder, general_head, scaler, config) を復元して返す（下流が呼ぶ）。

    encoder / general_head は eval・凍結済み。config で同一構成に再構築する。
    """
    d = _seed_dir(project, cutoff, seed_idx)
    with open(d / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    device = st.resolve_device() if device is None else device

    encoder = st.SetEncoder(config["d_model"], config["n_layers"], config["n_heads"],
                            config["ffn_dim"], config["dropout"]).to(device)
    encoder.load_state_dict(torch.load(d / "encoder.pt", map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    head = st.Head(config["d_model"], config["head_type"], config["head_hidden"]).to(device)
    head.load_state_dict(torch.load(d / "general_head.pt", map_location=device))
    head.eval()

    with open(d / "scaler.json", "r", encoding="utf-8") as f:
        sc = json.load(f)
    scaler = st.Scaler(np.array(sc["mean"], dtype=np.float64), np.array(sc["std"], dtype=np.float64))
    return encoder, head, scaler, config
