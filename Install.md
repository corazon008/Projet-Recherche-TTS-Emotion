## How to run

²### Build env

You can build an environment with `uv`.

#### To set up environment with uv

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

### Download and preprocess data

We used data of 10 English Speakers from [ESD dataset](https://github.com/HLTSingapore/Emotional-Speech-Data). To download all `.wav`, `.txt` files along with `.TextGrid` files created using [MFA](https://github.com/MontrealCorpusTools/mfa-models):

```bash
uv run --extra cpu download_data.py
```

To train a model we need precomputed durations, energy, pitch and eGeMap features. From `src` directory run:

```bash
uv run --extra cpu -m src.preprocess.preprocess
```

This is how your `app` folder should look like:

```
.
└── data
    ├── data
    │   └── ssw_esd
    ├── emospeech.ckpt
    ├── g_01800000
    ├── phones.json
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
    ├── ssw_esd.zip
    ├── test_ids.txt
    └── val_ids.txt
```

### Training

1. Configure arguments in `config/config.py`.
2. Run:

```bash
uv run --extra cpu -m src.scripts.train
```

### Testing

Testing is implemented on testing subset of ESD dataset. To synthesize audio and compute neural MOS (NISQA TTS):

1. Configure arguments in `config/config.py` under `Inference` section.
2. Run :

```bash
uv run --extra cpu -m src.scripts.test
```

You can find NISQA TTS for original, reconstructed and generated audio in `test.log`.

### Inference

EmoSpeech is trained on phoneme sequences. Supported phones can be found in `app/data/preprocessed/phones.json`. This repositroy is created for academic research and doesn't support automatic grapheme-to-phoneme conversion. However, if you would like to synthesize arbitrary sentence with emotion conditioning you can:

#### Using my custom docker image with MFA

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

#### Following the install guide of MFA

1. Generate phoneme sequence from graphemes with [MFA](https://github.com/MontrealCorpusTools/mfa-models).

   1.1 Follow the [installation guide](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

2. Download english g2p model: `mfa model download g2p english_us_arpa`

3. Generate phoneme.txt from graphemes.txt: `mfa g2p graphemes.txt english_us_arpa phoneme.txt`

#### Launch inference

Run `uv run --extra cpu -m src.scripts.inference`, specifying arguments:

| **Аrgument** | **Meaning**                          | **Possible Values**                                                 | **Default value**                              |
| ------------ | ------------------------------------ | ------------------------------------------------------------------- | ---------------------------------------------- |
| `-sq`        | Phoneme sequence to synthesisze      | Find in `data/phones.json`.                                         | **Not set, required argument if -pf not set.** |
| `-pf`        | Phoneme sequence file to synthesisze | `app/data/phoneme.txt`.                                             | **Not set, required argument if -sq not set.** |
| `-emo`       | Id of desired voice emotion          | 0: neutral, 1: angry, 2: happy, 3: sad, 4: surprise.                | 1                                              |
| `-sp`        | Id of speaker voice                  | From 1 to 10, correspond to 0011 ... 0020 in original ESD notation. | 5                                              |
| `-p`         | Path where to save synthesised audio | Any with `.wav` extension.                                          | generation_from_phoneme_sequence.wav           |

For example

```bash
uv run --extra cpu -m src.scripts.inference -sq "S P IY2 K ER1 F AY1 V  T AO1 K IH0 NG W IH0 TH AE1 NG G R IY0 IH0 M OW0 SH AH0 N"
```

```bash
uv run --extra cpu -m src.scripts.inference -pf app/data/phoneme.txt
```

If result file is not synthesied, check `inference.log` for OOV phones.
