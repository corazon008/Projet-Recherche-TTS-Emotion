# How to run

## Build env

You can build an environment with `uv`.

### To set up environment with uv

If you don't have uv installed, please find the installation instructions for your OS [here](https://docs.astral.sh/uv/getting-started/installation/).

`uv` will create a virtual environment and install all dependencies specified in `pyproject.toml`. To build env with `uv` run:

For CPU-only:

```bash
uv sync --extra cpu
```

For GPU:

```bash
uv sync --extra gpu
```

## Download and preprocess data

We used data of 10 English Speakers from [ESD dataset](https://github.com/HLTSingapore/Emotional-Speech-Data). To download all `.wav`, `.txt` files along with `.TextGrid` files created using [MFA](https://github.com/MontrealCorpusTools/mfa-models):

```bash
uv run --extra cpu download_data.py
```

To train a model we need precomputed durations, energy, pitch and eGeMap features.

```bash
uv run --extra cpu -m src.preprocess.preprocess
```

This is how your `app` folder should look like:

```
.
└── data
    ├── data
    │   └── ssw_esd
    ├── emospeech.cpkt
    ├── preprocessed
    │   ├── duration
    │   ├── egemap
    │   ├── energy
    │   ├── mel
    │   ├── phones.json
    │   ├── pitch
    │   ├── stats.json
    │   ├── test.txt
    │   ├── train.txt
    │   ├── trimmed_wav
    │   └── val.txt
    ├── test_ids.txt
    ├── val_ids.txt
    └── vocoder_checkpoint.pt
```

## Training

Two conditions are supported with the **same architecture**, differing only in the
emotion conditioning (this is the ablation used by the project):

- `--use-emotion` → emotion embeddings are used.
- `--no-use-emotion` → baseline: only text + speaker, emotion information removed.

1. Defaults live in `config/config.py`. Any `TrainConfig` field can be overridden on
   the command line with `--field value` (e.g. `--train_batch_size 48`).
2. Train the **emotion** model:

```bash
uv run --extra gpu -m src.scripts.train \
  --use-emotion --device cuda \
  --train_batch_size 48 \
  --nb_epochs 2000 \
  --total_training_steps 300000 \
  --val_each_epoch 100 \
  --num_workers 8
```

3. Train the **baseline (no emotion)** with a strictly identical protocol:

```bash
uv run --extra gpu -m src.scripts.train \
  --no-use-emotion --device cuda \
  --train_batch_size 48 \
  --nb_epochs 2000 \
  --total_training_steps 300000 \
  --val_each_epoch 100 \
  --num_workers 8
```

Notes:

- Checkpoints go to `app/data/checkpoints/with_emotion/` and
  `app/data/checkpoints/no_emotion/` automatically. Best models are selected by
  `val_mos/generated_audio_mos_mean` (NISQA); `last.ckpt` is always written.
- Training stops at whichever comes first: `nb_epochs` or `total_training_steps`
  (used as `max_steps`). To keep the comparison valid, run both conditions with the
  same seed (`config.seed`, default 3), batch size and step budget.
- Validation is expensive (NISQA MOS + audio logging) and runs every
  `val_each_epoch` epochs — use a large value (e.g. 100) for long runs.
- Resuming from the other condition is refused. Resume an interrupted run like this:

```bash
uv run --extra gpu -m src.scripts.train --use-emotion --device cuda \
  --train_from_checkpoint last.ckpt --nb_epochs 2000 --total_training_steps 300000
```

## Testing

Testing synthesizes the test subset of the ESD dataset and computes NISQA TTS MOS for
original, reconstructed and generated audio. The modality (emotion / no emotion) is
read automatically from the checkpoint config:

```bash
uv run --extra gpu -m src.scripts.test \
  --checkpoint app/data/checkpoints/with_emotion/last.ckpt --device cuda
```

You can find NISQA TTS for original, reconstructed and generated audio in `test.log`.
Synthesized wav files are written under `config.audio_save_path`
(default `app/data/deepvk_test`).

Optional flags: `--no-use-emotion` / `--use-emotion` force the modality (overriding
the checkpoint config), `--device cuda|cpu` selects the device.

## Inference

EmoSpeech is trained on phoneme sequences. Supported phones can be found in `app/data/preprocessed/phones.json`. This repository is created for academic research and doesn't support automatic grapheme-to-phoneme conversion. However, if you would like to synthesize arbitrary sentence with emotion conditioning you can:

### Using my custom docker image with MFA

1. Build the docker image with MFA:

```bash
docker build -t mfa .
```

2. Run the docker container, mounting `app/data` to `/data` in the container:

```bash
docker run -it -v ./app/data:/data mfa
```

3. Create `graphemes.txt` file in `app/data` with the sentence you want to synthesize, for example:

```bash
echo "Your sentence to synthesize goes here." > app/data/graphemes.txt
```

### Following the install guide of MFA

1. Generate phoneme sequence from graphemes with [MFA](https://github.com/MontrealCorpusTools/mfa-models).

    1.1 Follow the [installation guide](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

2. Download english g2p model: `mfa model download g2p english_us_arpa`

3. Generate phoneme.txt from graphemes.txt: `mfa g2p graphemes.txt english_us_arpa phoneme.txt`

### Launch inference

The modality used for synthesis is the one of the checkpoint (auto-detected from its
config). For a no-emotion model the `-emo` argument is ignored.

Run `uv run --extra gpu -m src.scripts.inference`, specifying arguments:

| **Argument**   | **Meaning**                         | **Possible Values**                                  | **Default value**                       |
| -------------- | ----------------------------------- | ---------------------------------------------------- | --------------------------------------- |
| `--checkpoint` | Trained checkpoint to use           | `app/data/checkpoints/with_emotion/last.ckpt` etc.   | `config.testing_checkpoint`             |
| `--device`     | Device to run on                    | `cuda`, `cpu`                                        | `cuda` if available, else `cpu`         |
| `-sq`          | Phoneme sequence to synthesize      | Find in `app/data/preprocessed/phones.json`.         | **Not set, required if `-pf` not set.** |
| `-pf`          | Phoneme sequence file to synthesize | `app/data/phoneme.txt`.                              | **Not set, required if `-sq` not set.** |
| `-emo`         | Id of desired emotion               | 0: neutral, 1: angry, 2: happy, 3: sad, 4: surprise. | 1                                       |
| `-sp`          | Id of speaker voice                 | From 1 to 10 (0011 ... 0020 in ESD notation).        | 5                                       |
| `-p`           | Path where to save the audio        | Any `.wav` path.                                     | generation_from_phoneme_sequence.wav    |

Optional flags: `--use-emotion` / `--no-use-emotion` force the modality.

For example

```bash
uv run --extra gpu -m src.scripts.inference \
  --checkpoint app/data/checkpoints/with_emotion/last.ckpt \
  -sq "S P IY2 K ER1 F AY1 V  T AO1 K IH0 NG W IH0 TH AE1 NG G R IY0 IH0 M OW0 SH AH0 N"
```

```bash
uv run --extra gpu -m src.scripts.inference \
  --checkpoint app/data/checkpoints/no_emotion/last.ckpt \
  -pf app/data/phoneme.txt
```

If the result file is not synthesized, check `inference.log` for OOV phones.
