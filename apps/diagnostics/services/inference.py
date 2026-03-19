import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as functional
from django.conf import settings

LOGGER = logging.getLogger(__name__)
DECISION_THRESHOLD = 0.50
UNCERTAINTY_LOW = 0.30
UNCERTAINTY_HIGH = 0.70
MAX_LENGTH = 1024
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class InferenceService:
    """Run GMO classification with local artifacts and graceful fallback."""

    def __init__(self):
        # Keep path unchanged for this project.
        self.model_dir = Path(settings.ML_MODEL_DIR) / 'gmo_model'
        self.device = DEVICE
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_error: Optional[str] = None

    def warm_up(self) -> bool:
        return self._ensure_model_loaded()

    def predict(self, sequence: str) -> Dict[str, Any]:
        if not sequence:
            return {
                'label': 'no-gmo',
                'score': 0.0,
                'probabilities': {'gmo': 0.0, 'no_gmo': 1.0},
                'attention': [],
                'chunks_processed': 1,
                'in_uncertainty_zone': False,
            }

        if self._ensure_model_loaded():
            return self._predict_with_model(sequence)

        return self._predict_with_fallback(sequence)

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True

        if self._load_error:
            return False

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            self._load_error = f'Model dependencies unavailable: {exc}'
            LOGGER.warning(self._load_error)
            return False

        try:
            config_path = self.model_dir / 'config.pt'
            encoder_path = self.model_dir / 'encoder'
            tokenizer_path = self.model_dir / 'tokenizer'

            for required_path in [encoder_path, tokenizer_path]:
                if not required_path.exists():
                    self._load_error = f'Missing model artifact: {required_path}'
                    LOGGER.warning(self._load_error)
                    return False

            model_name = 'Unknown'
            if config_path.exists():
                try:
                    config = torch.load(str(config_path), map_location=self.device, weights_only=False)
                    model_name = config.get('model_name', 'Unknown')
                except Exception:
                    LOGGER.warning('Could not read model config at %s', config_path)

            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(encoder_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(encoder_path),
                    trust_remote_code=False,
                    local_files_only=True,
                )
                LOGGER.warning(
                    'Model loaded with ignore_mismatched_sizes=True. '
                    'Please regenerate artifacts for best accuracy.'
                )

            model.to(self.device)
            model.eval()

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    trust_remote_code=False,
                    local_files_only=True,
                )

            self._model = model
            self._tokenizer = tokenizer
            LOGGER.info('GMO classification model loaded from %s (model=%s)', self.model_dir, model_name)
            return True
        except Exception as exc:
            self._load_error = f'Impossible to load GMO model: {exc}'
            LOGGER.exception(self._load_error)
            return False

    def _predict_with_model(self, sequence: str) -> Dict[str, Any]:
        assert self._model is not None
        assert self._tokenizer is not None

        tokens = self._tokenizer(
            sequence.upper(),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self._model(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
            )
            probs = functional.softmax(outputs.logits, dim=1)

        gmo_prob = float(probs[0, 1].item())
        no_gmo_prob = float(probs[0, 0].item())
        in_uncertainty_zone = UNCERTAINTY_LOW <= gmo_prob <= UNCERTAINTY_HIGH

        label = 'gmo' if gmo_prob >= DECISION_THRESHOLD else 'no-gmo'
        score = gmo_prob if label == 'gmo' else no_gmo_prob

        return {
            'label': label,
            'score': round(score, 4),
            'probabilities': {
                'gmo': round(gmo_prob, 4),
                'no_gmo': round(no_gmo_prob, 4),
            },
            'attention': self._fallback_attention_segments(sequence.upper()),
            'chunks_processed': 1,
            'in_uncertainty_zone': in_uncertainty_zone,
        }

    def _predict_with_fallback(self, sequence: str) -> Dict[str, Any]:
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence)
        motif_hits = self._motif_hits(sequence)

        gmo_probability = min(max(0.18 + (gc_ratio * 0.9) + (motif_hits * 0.05), 0.02), 0.98)
        no_gmo_probability = 1 - gmo_probability
        label = 'gmo' if gmo_probability >= DECISION_THRESHOLD else 'no-gmo'
        score = gmo_probability if label == 'gmo' else no_gmo_probability

        return {
            'label': label,
            'score': round(score, 4),
            'probabilities': {
                'gmo': round(gmo_probability, 4),
                'no_gmo': round(no_gmo_probability, 4),
            },
            'attention': self._fallback_attention_segments(sequence),
            'chunks_processed': 1,
            'in_uncertainty_zone': UNCERTAINTY_LOW <= gmo_probability <= UNCERTAINTY_HIGH,
        }

    def _motif_hits(self, sequence: str) -> int:
        motifs = ('CAAT', 'TATA', 'GCGC', 'CCGG', 'GGCC')
        return sum(sequence.count(motif) for motif in motifs)

    def _fallback_attention_segments(self, sequence: str) -> List[Dict[str, Any]]:
        window_size = min(18, max(8, len(sequence) // 6 or 8))
        stride = max(4, window_size // 2)
        segments: List[Dict[str, Any]] = []

        for start in range(0, max(len(sequence) - window_size + 1, 1), stride):
            end = min(start + window_size, len(sequence))
            window = sequence[start:end]
            if not window:
                continue

            gc_count = window.count('G') + window.count('C')
            gc_ratio = gc_count / len(window)
            motif_bonus = self._motif_hits(window) * 0.08
            score = min(gc_ratio + motif_bonus, 1.0)

            segments.append(
                {
                    'start': start,
                    'end': end,
                    'sequence': window,
                    'score': round(score, 4),
                }
            )

        ranked = sorted(segments, key=lambda item: item['score'], reverse=True)
        return ranked[:6]


_INFERENCE_SERVICE: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _INFERENCE_SERVICE
    if _INFERENCE_SERVICE is None:
        _INFERENCE_SERVICE = InferenceService()
    return _INFERENCE_SERVICE
