import json
import numpy as np

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from backend.services.preprocessing import preprocess_text


# Project paths
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


class SentimentPredictor:

    def __init__(self):
        # Load trained model
        self.model = load_model(ARTIFACTS / "sentiment_model.keras")

        # Load tokenizer
        with open(ARTIFACTS / "tokenizer.json", "r") as f:
            self.tokenizer = tokenizer_from_json(f.read())

        # Load configuration
        with open(ARTIFACTS / "config.json", "r") as f:
            self.config = json.load(f)

        self.max_length = self.config["max_length"]
        self.classes = self.config["classes"]

    def predict(self, text):
        # Apply same preprocessing used during training
        cleaned_text = preprocess_text(text)

        if not cleaned_text.strip():
            raise ValueError("Comment contains no usable text.")

        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        padded = pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding="post"
        )

        # Predict
        probabilities = self.model.predict(padded, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))

        return {
            "sentiment": self.classes[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                label: float(probabilities[i])
                for i, label in enumerate(self.classes)
            }
        }