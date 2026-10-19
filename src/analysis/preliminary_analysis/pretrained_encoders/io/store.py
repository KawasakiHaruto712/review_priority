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

from src.analysis.preliminary_analysis.pretrained_encoders.features.feature_set import MODEL_FEATURES
from src.analysis.preliminary_analysis.pretrained_encoders.model import set_transformer as st
from src.analysis.preliminary_analysis.pretrained_encoders.utils import constants


def _seed_dir(project: str, cutoff: str, seed_idx: int) -> Path:
    return constants.OUTPUT_ROOT / project / f"cutoff_{cutoff}" / f"seed{seed_idx}"


def _base_config(project: str, cutoff: str, seed_idx: int) -> dict:
    return {
        "project": project, "cutoff": cutoff, "seed": seed_idx,
        "feature_names": list(MODEL_FEATURES), "feature_dim": len(MODEL_FEATURES),   # モデルの入力の並び（§6・§12.5）
        "d_model": constants.D_MODEL, "n_layers": constants.N_LAYERS, "n_heads": constants.N_HEADS,
        "ffn_dim": constants.FFN_DIM, "dropout": constants.DROPOUT, "max_set_size": constants.MAX_SET_SIZE,
        "head_type": constants.HEAD_TYPE, "head_hidden": constants.HEAD_HIDDEN,
        "pretrain_method": constants.PRETRAIN_METHOD, "pretrain_epochs": constants.PRETRAIN_EPOCHS,
        "pretrain_lr": constants.PRETRAIN_LR, "pretrain_batch_sets": constants.PRETRAIN_BATCH_SETS,
        "pretrain_micro_sets": constants.PRETRAIN_MICRO_SETS,   # 勾配累積（design.md §4。2026-10-19 から。これより前の保存にはこの項目がない）
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


class FeatureMismatchError(RuntimeError):
    """保存したエンコーダの特徴量の並びが、今の MODEL_FEATURES と違う（design.md §6）。"""


def check_feature_names(config: dict, project: str, cutoff: str) -> None:
    """保存した config の feature_names が今の MODEL_FEATURES と（並びも含めて）違えば止める（design.md §6。2026-10-18）。"""
    saved = list(config.get("feature_names") or [])
    if saved != list(MODEL_FEATURES):
        only_saved = [n for n in saved if n not in MODEL_FEATURES]
        only_now = [n for n in MODEL_FEATURES if n not in saved]
        raise FeatureMismatchError(
            f"[{project} cutoff={cutoff}] 保存したエンコーダの特徴量（{len(saved)} 個）が今の MODEL_FEATURES（{len(MODEL_FEATURES)} 個）と違います"
            f"（保存にだけある: {only_saved}、今にだけある: {only_now}{'、並びが違う' if not only_saved and not only_now else ''}）。"
            f"事前学習を作り直してください：python3 -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders "
            f"--project {project}（古いエンコーダは別の場所へ退避してから）")


class NonFiniteWeightsError(RuntimeError):
    """保存したエンコーダ・汎用ヘッドの重みか Scaler に NaN・無限大がある（design.md §6。2026-10-20）。"""


def check_finite_weights(encoder, head, scaler, project: str, cutoff: str, seed_idx: int) -> None:
    """重みと Scaler に NaN・無限大があれば止める（design.md §6。qtbase の seed0〜2 の NaN の重みを下流が黙って読めたため）。"""
    bad = [f"encoder.{n}" for n, t in encoder.state_dict().items() if not torch.isfinite(t).all()]
    bad += [f"general_head.{n}" for n, t in head.state_dict().items() if not torch.isfinite(t).all()]
    bad += [f"scaler.{k}" for k, v in (("mean", scaler.mean), ("std", scaler.std)) if not np.all(np.isfinite(v))]
    if bad:
        raise NonFiniteWeightsError(
            f"[{project} cutoff={cutoff} seed{seed_idx}] 保存したエンコーダに NaN・無限大があります（{len(bad)} か所：{bad[:5]}）。"
            f"事前学習を作り直してください：python3 -m src.analysis.preliminary_analysis.pretrained_encoders.build_encoders "
            f"--project {project}（古いエンコーダは別の場所へ退避してから）")


def save_nonfinite_record(project: str, cutoff: str, seed_idx: int, info: dict) -> Path:
    """事前学習が非有限で止まった seed の記録を書く（design.md §4 の NaN の点検。モデルは保存しない）。"""
    base = constants.OUTPUT_ROOT / project / f"cutoff_{cutoff}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"seed{seed_idx}_nonfinite.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"project": project, "cutoff": cutoff, "seed": seed_idx, **info}, f, ensure_ascii=False, indent=2)
    return path


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
    check_feature_names(config, project, cutoff)
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
    check_finite_weights(encoder, head, scaler, project, cutoff, seed_idx)   # 2026-10-20（§6）
    return encoder, head, scaler, config
