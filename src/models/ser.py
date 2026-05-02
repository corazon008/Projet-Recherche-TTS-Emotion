from funasr import AutoModel

# emotion2vec+ model's classes
EMOTION_LABELS = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
    8: "unknown",
}

# target's emotions
TARGET_EMOTIONS = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprised",
}


class SER:
    def __init__(self):
        self.model = AutoModel(
            model="iic/emotion2vec_plus_large",
            hub="ms",
            disable_update=True
        )

    def _find_label_index(self, labels: list, target: str) -> int:
        """Trouve l'index d'un label même au format '中文/english'"""
        for i, label in enumerate(labels):
            if target.lower() in label.lower():
                return i
        raise ValueError(f"Émotion '{target}' introuvable dans les labels : {labels}")

    def check(self, wav_path: str, expected_emotion: str) -> dict:
        """
        Check if audio file has the expected emotion
        """
        result = self.model.generate(
            wav_path,
            granularity="utterance",
            extract_embedding=False,
        )

        print("result brut:", result)  # debug

        scores = result[0]["scores"]
        labels = result[0]["labels"]

        print("labels:", labels)  # debug
        print("scores:", scores)  # debug

        # Expected emotion score
        target_label = TARGET_EMOTIONS[expected_emotion]
        target_idx = self._find_label_index(labels, target_label)
        target_score = scores[target_idx]

        # Predicted emotion
        predicted_idx = scores.index(max(scores))
        predicted_label = labels[predicted_idx]
        is_match = target_label.lower() in predicted_label.lower()

        return {
            "expected": expected_emotion,
            "predicted": predicted_label,
            "match": is_match,
            "target_score": round(target_score, 4),
            "all_scores": dict(zip(labels, [round(s, 4) for s in scores])),
        }