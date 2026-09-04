import json
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger
from scipy.io import wavfile

from config.config import TrainConfig


def config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[TrainConfig]:
    """Return the TrainConfig stored in a Lightning checkpoint hyper_parameters."""
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    config = hyper_parameters.get("config", None)
    if isinstance(config, dict):
        config = TrainConfig(**{k: v for k, v in config.items() if k in {f.name for f in fields(TrainConfig)}})
    return config


def build_config_from_checkpoint(
    checkpoint_path: Union[str, Path], **overrides
) -> TrainConfig:
    """Config used for testing/inference: starts from current defaults, then applies the
    stored checkpoint config (so a no-emotion model is never evaluated in emotion mode),
    then explicit CLI overrides."""
    config = TrainConfig()
    stored = config_from_checkpoint(checkpoint_path)
    if stored is not None:
        for field in fields(config):
            if hasattr(stored, field.name):
                setattr(config, field.name, getattr(stored, field.name))
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def compute_overall_mos(d: dict) -> tuple[float, float]:
    return np.mean(list(d.values())), np.std(list(d.values()))


def compute_mos_per_speaker(d: dict) -> dict:
    res = {}
    res = defaultdict(lambda: [], res)
    for basename, score in zip(list(d.keys()), list(d.values())):
        speaker, _, _ = basename.split("_")
        res[speaker].append(score)
    mos_dict = {}
    for k, v in zip(list(res.keys()), list(res.values())):
        mos_dict[k] = np.mean(v)
    return mos_dict


def write_txt(txt_path: Path, data: list) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        for m in data:
            f.write(m + "\n")


def set_up_logger(filename: str) -> None:
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    logger.add(filename, format=fmt)


def crash_with_msg(message: str) -> None:
    logger.error(message)
    sys.exit(1)


def write_wav(path: Union[Path, str], wav: np.ndarray, sample_rate=16000) -> None:
    wavfile.write(path, sample_rate, wav)


def write_json(d: dict, path: Union[Path, str]) -> None:
    with open(path, "a") as f:
        json.dump(d, f)
        f.write("\n")
