import whisper
import torch
from loguru import logger

class STTModel:
    """stt avec whisper"""
    
    def __init__(self, model_size="base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        #logger.info(f"Load Whisper '{model_size}' on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def get_text(self, audio, language=None, **args):
        """
            audio: Path for the audio file
            language: optional langage code (fr,en,...)
            **args: more options for whisper
        """
        options = {
            "language": language,
            "fp16": self.device == "cuda",
            **args
        }
        
        result = self.model.transcribe(audio, **options)
        return result["text"].strip()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Do: python stt.py <chemin>")
        sys.exit(1)

    stt = STTModel()
    texte = stt.get_text(sys.argv[1])
    print(f"\nText : {texte}")
