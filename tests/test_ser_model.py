from src.models.ser import SER
from pathlib import Path

def test_happy_emotion():
    checker = SER()

    base_dir = Path(__file__).parent.parent
    name_file = "0011_000702.wav"
    wav_path = base_dir / "datasets" / "Emotion Speech Dataset" / "0011" / "Happy" / name_file
    assert wav_path.exists(), f"Could not find file : {wav_path}"

    result = checker.check(str(wav_path), expected_emotion="happy")
    print(result)

    assert result["match"] is True
    assert result["target_score"] > 0.3