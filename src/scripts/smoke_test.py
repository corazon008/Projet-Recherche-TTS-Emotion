import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path

import lightning
import torch
from lightning import seed_everything

from config.config import TrainConfig
from src.dataset.dataset import get_dataloader
from src.models import Generator, TorchSTFT
from src.models.acoustic_model.fastspeech.lightning_model import (
    FastSpeechLightning,
)
from src.utils.utils import build_config_from_checkpoint
from src.utils.vocoder_utils import load_checkpoint


def _make_model(cfg):
    vocoder = Generator(**asdict(cfg))
    vocoder.load_state_dict(load_checkpoint(cfg.vocoder_checkpoint_path)["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    stft = TorchSTFT(**asdict(cfg))
    model = FastSpeechLightning(cfg, vocoder, stft, train=False)
    model.eval()
    return model


def _forward(model, loader):
    with torch.no_grad():
        for batch in loader:
            batch_dict = model._get_batch_dict_from_dataloader(
                batch, validation=True
            )
            output = model.model(torch.device("cpu"), batch_dict)
            return output["predicted_mel"].clone(), output["mel_len"].clone()


def check_reload_consistency(cfg, tmp_dir):
    model = _make_model(cfg)
    loader = get_dataloader(cfg, "val", size=8)
    mel_a, lens_a = _forward(model, loader)
    ckpt_path = Path(tmp_dir) / "model.ckpt"
    ckpt = {
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": lightning.__version__,
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "loops": {},
        "callbacks": {},
    }
    torch.save(ckpt, str(ckpt_path))

    reloaded_cfg = build_config_from_checkpoint(ckpt_path, device="cpu")
    assert (
        reloaded_cfg.use_emotion_embeddings == cfg.use_emotion_embeddings
    ), "checkpoint modality not preserved"

    vocoder = Generator(**asdict(reloaded_cfg))
    vocoder.load_state_dict(
        load_checkpoint(reloaded_cfg.vocoder_checkpoint_path)["generator"]
    )
    vocoder.remove_weight_norm()
    vocoder.eval()
    stft = TorchSTFT(**asdict(reloaded_cfg))
    reloaded = FastSpeechLightning.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        strict=True,
        config=reloaded_cfg,
        vocoder=vocoder,
        stft=stft,
        train=False,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    reloaded.eval()
    mel_b, lens_b = _forward(reloaded, loader)

    assert torch.allclose(lens_a, lens_b), "mel lengths differ after reload"
    max_diff = (mel_a - mel_b).abs().max().item()
    assert max_diff < 1e-4, f"mel outputs differ after reload: max diff {max_diff}"
    print(
        f"OK modality={cfg.use_emotion_embeddings}: reload preserves "
        f"config & outputs (max diff {max_diff:.2e})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    seed_everything(3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for use_emotion in (True, False):
            cfg = TrainConfig()
            cfg.device = "cpu"
            cfg.num_workers = 0
            cfg.train_batch_size = 4
            cfg.val_batch_size = 4
            cfg.use_emotion_embeddings = use_emotion
            check_reload_consistency(cfg, tmp_dir)


if __name__ == "__main__":
    main()
