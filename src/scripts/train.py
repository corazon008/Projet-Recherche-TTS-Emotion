from dataclasses import asdict
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from config.config import TrainConfig
from src.dataset.dataset import get_dataloader
from src.models import Generator, TorchSTFT
from src.models.acoustic_model.fastspeech.lightning_model import (
    FastSpeechLightning,
)
from src.utils.utils import config_from_checkpoint, crash_with_msg, set_up_logger
from src.utils.vocoder_utils import load_checkpoint


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    vocoder = Generator(**asdict(config))
    stft = TorchSTFT(**asdict(config))
    vocoder_state_dict = load_checkpoint(config.vocoder_checkpoint_path)
    vocoder.load_state_dict(vocoder_state_dict["generator"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    train_loader = get_dataloader(config, "train")
    val_loader = get_dataloader(config, "val")
    test_loader = get_dataloader(config, "test")
    model = FastSpeechLightning(config, vocoder, stft)

    Path(config.lightning_checkpoint_path).mkdir(exist_ok=True, parents=True)
    callbacks = ModelCheckpoint(
        dirpath=config.lightning_checkpoint_path,
        filename="fs-{epoch}-{total_loss:.3f}",
        monitor="val_mos/generated_audio_mos_mean",
        save_top_k=config.save_top_k_model_weights,
        mode=config.metric_monitor_mode,
        save_last=True,
        every_n_epochs=4,
        enable_version_counter=True,
    )

    progress_bar = TQDMProgressBar(
        refresh_rate=config.wandb_progress_bar_refresh_rate
    )
    # wandb_logger.watch(model.model, log_graph=False)

    trainer = Trainer(
        max_steps=config.total_training_steps,
        max_epochs=config.nb_epochs,
        check_val_every_n_epoch=config.val_each_epoch,
        log_every_n_steps=config.wandb_log_every_n_steps,
        # logger=wandb_logger,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=list(config.devices) if config.devices else "auto",
        callbacks=[callbacks, progress_bar],
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        num_sanity_val_steps=config.num_sanity_val_steps,
        strategy=config.strategy,
        deterministic=True,
        enable_checkpointing=True,
        precision=config.precision,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)

    resume_path = None
    if config.train_from_checkpoint:
        resume_path = Path(config.lightning_checkpoint_path) / config.train_from_checkpoint
        stored_config = config_from_checkpoint(resume_path)
        if (
            stored_config is not None
            and stored_config.use_emotion_embeddings != config.use_emotion_embeddings
        ):
            crash_with_msg(
                "Cannot resume: checkpoint was trained "
                f"use_emotion_embeddings={stored_config.use_emotion_embeddings} "
                f"but the current run uses use_emotion_embeddings={config.use_emotion_embeddings}. "
                "Train each modality from scratch."
            )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_path,
        weights_only=config.weights_only,
    )
    trainer.validate(model, dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


def apply_unknown_args(config, unknown):
    it = iter(unknown)

    for arg in it:
        if not arg.startswith("--"):
            continue

        key = arg[2:]
        value = next(it)

        current = getattr(config, key)

        # cast automatique basé sur le type existant
        target_type = type(current)

        if target_type is bool:
            value = value.lower() in ("1", "true", "yes")
        else:
            value = target_type(value)

        setattr(config, key, value)


if __name__ == "__main__":
    import argparse

    set_up_logger("train.log")

    parser = argparse.ArgumentParser(description="Train FastSpeech2 model")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-emotion",
        dest="use_emotion",
        action="store_true",
        help="enable emotion embeddings (overrides config.use_emotion_embeddings)",
    )
    group.add_argument(
        "--no-use-emotion",
        dest="use_emotion",
        action="store_false",
        help="disable emotion embeddings (overrides config.use_emotion_embeddings)",
    )
    parser.set_defaults(use_emotion=None)

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config fields: key=value",
    )

    args, unknown = parser.parse_known_args()

    config = TrainConfig()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert args.use_emotion is not None
    config.use_emotion_embeddings = args.use_emotion
    if args.use_emotion:
        config.lightning_checkpoint_path /= "with_emotion"
    else:
        config.lightning_checkpoint_path /= "no_emotion"

    apply_unknown_args(config, unknown)

    train(config)
