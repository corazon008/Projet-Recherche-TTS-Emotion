import torchaudio
from functools import cache

from torch.utils.data import Dataset
import librosa
from pathlib import Path
import os
from typing import Union, List, Tuple
import pandas as pd
import numpy as np

from utils import find_project_root


class ESDDataset(Dataset):
    def __init__(self, path: Union[str, Path]):
        self.path = path
        self.speakers = sorted(os.listdir(path))
        self.data: List[Tuple[str, str, str, str]] = []
        for speaker in self.speakers:
            # Check if speaker ID is in range 11-20 because the others are not english speakers
            if not int(speaker) in range(11, 21):
                continue
            speaker_path = os.path.join(path, speaker)
            for emotion in os.listdir(speaker_path):
                if os.path.isdir(os.path.join(speaker_path, emotion)):
                    for file in os.listdir(os.path.join(speaker_path, emotion)):
                        df = self.get_transcript(speaker)
                        line = df[df.iloc[:, 0] == Path(file).stem]
                        text = line[1].values[0]
                        self.data.append(
                            (
                                speaker,
                                emotion,
                                text,
                                os.path.join(speaker_path, emotion, file),
                            )
                        )

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Tuple[np.ndarray, Union[int, float], str, str]:
        speaker, emotion, text, file_path = self.data[idx]
        # signal, sr = librosa.load(file_path, sr=None)
        signal, sr = torchaudio.load(file_path)
        return signal, sr, text, emotion

    @cache
    def get_transcript(self, speaker: str) -> pd.DataFrame:
        speaker_path = os.path.join(self.path, speaker)
        df = pd.read_csv(
            os.path.join(speaker_path, f"{speaker}.txt"), sep="\t", header=None
        )
        return df


if __name__ == "__main__":
    dataset = ESDDataset(
        find_project_root() / Path("datasets/Emotion Speech Dataset/")
    )
    print(f"Number of samples: {len(dataset)}")
    print(f"Speakers: {dataset.speakers}")
