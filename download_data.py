import gdown
import os
import zipfile
from pathlib import Path

# set working dir to app/data
wd = Path(__file__).parent / "app" / "data"
wd.mkdir(parents=True, exist_ok=True)
os.chdir(wd)

gdown.download(
    "https://drive.google.com/uc?id=1n29y5hr8JK2tGEdYNLtIXO8hWw7DkdJ1",
    "ssw_esd.zip",
    quiet=False,
)  # download ssw_esd.zip (wavs, txt, TextGrids)
gdown.download(
    "https://drive.google.com/uc?id=1F-_o_6x43IuL-81QHozAab7l3PPG2DNE",
    "val_ids.txt",
    quiet=False,
)  # download val_ids.txt
gdown.download(
    "https://drive.google.com/uc?id=1VgvuU1p79GLPikwGl3lO6lanDdneHZud",
    "test_ids.txt",
    quiet=False,
)  # download test_ids.txt
gdown.download(
    "https://drive.google.com/uc?id=1KqPK3A8JpB57pzy5dEgkX9RRRkkPZS7S",
    "vocoder_checkpoint",
    quiet=False,
)  # download vocoder_checkpoint
gdown.download(
    "https://drive.google.com/uc?id=1loIzPv4BGoXVYgDtBu_BXbKytiQoAjmW",
    "EmoSpeech_checkpoint",
    quiet=False,
)  # download EmoSpeech checkpoint
gdown.download(
    "https://drive.google.com/uc?id=1_g23gaPC7K1hO-kWJk9JOPJZR8olimxw",
    "phones.json",
    quiet=False,
)  # download phones.json

# unzip ssw_esd.zip
print("Unzipping ssw_esd.zip...")
with zipfile.ZipFile("ssw_esd.zip", "r") as zip_ref:
    zip_ref.extractall(".")
