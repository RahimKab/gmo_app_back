from pathlib import Path
from typing import Dict

from django.conf import settings


class InferenceService:
    """Simple placeholder service to keep prediction logic isolated."""

    def __init__(self):
        self.model_dir = Path(settings.ML_MODEL_DIR)

    def predict(self, sequence: str) -> Dict[str, float | str]:
        if not sequence:
            return {'label': 'invalid', 'score': 0.0}

        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence)

        label = 'likely_gmo' if gc_ratio >= 0.55 else 'likely_non_gmo'
        return {'label': label, 'score': round(float(gc_ratio), 4)}
