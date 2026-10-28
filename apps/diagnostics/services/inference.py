import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as functional
from django.conf import settings

LOGGER = logging.getLogger(__name__)
DECISION_THRESHOLD = 0.50
UNCERTAINTY_LOW = 0.30
UNCERTAINTY_HIGH = 0.70
MAX_LENGTH = 512
INFERENCE_CHUNK_SIZE = 2048
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class InferenceService:
    """Run GMO classification with local artifacts and graceful fallback."""

    def __init__(self):
        self.model_dir = Path(settings.ML_MODEL_DIR) / 'gmo_model'
        self.encoder_dir = self.model_dir / 'encoder'
        self.classifier_path = self.model_dir / 'classifier_head.pt'
        self.device = DEVICE
        self._encoder: Optional[Any] = None
        self._classifier: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._load_error: Optional[str] = None

    def warm_up(self) -> bool:
        return self._ensure_model_loaded()

    def get_load_status(self) -> Dict[str, Any]:
        return {
            'loaded': (
                self._encoder is not None
                and self._classifier is not None
                and self._tokenizer is not None
            ),
            'device': self.device,
            'model_dir': str(self.model_dir),
            'error': self._load_error,
        }

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
            predictor = self._predict_with_model
        else:
            predictor = self._predict_with_fallback

        if len(sequence) <= INFERENCE_CHUNK_SIZE:
            return predictor(sequence)

        return self._predict_in_chunks(sequence, predictor)

    def _predict_in_chunks(
        self,
        sequence: str,
        predictor: Any,
    ) -> Dict[str, Any]:
        chunks = [
            sequence[start:start + INFERENCE_CHUNK_SIZE]
            for start in range(0, len(sequence), INFERENCE_CHUNK_SIZE)
        ]
        predictions = [predictor(chunk) for chunk in chunks]
        total_length = len(sequence)
        gmo_probability = sum(
            result['probabilities']['gmo'] * len(chunk)
            for result, chunk in zip(predictions, chunks)
        ) / total_length
        no_gmo_probability = 1 - gmo_probability
        reported_gmo_probability = round(gmo_probability, 4)
        reported_no_gmo_probability = round(no_gmo_probability, 4)
        label = 'gmo' if gmo_probability >= DECISION_THRESHOLD else 'no-gmo'
        score = gmo_probability if label == 'gmo' else no_gmo_probability

        attention: List[Dict[str, Any]] = []
        if reported_gmo_probability > 0:
            for offset, result in zip(range(0, len(sequence), INFERENCE_CHUNK_SIZE), predictions):
                for segment in result['attention']:
                    attention.append(
                        {
                            **segment,
                            'start': segment['start'] + offset,
                            'end': segment['end'] + offset,
                        }
                    )

        return {
            'label': label,
            'score': round(score, 4),
            'probabilities': {
                'gmo': reported_gmo_probability,
                'no_gmo': reported_no_gmo_probability,
            },
            'attention': sorted(attention, key=lambda item: item['score'], reverse=True),
            'chunks_processed': len(chunks),
            'in_uncertainty_zone': UNCERTAINTY_LOW <= gmo_probability <= UNCERTAINTY_HIGH,
        }

    def _ensure_model_loaded(self) -> bool:
        if (
            self._encoder is not None
            and self._classifier is not None
            and self._tokenizer is not None
        ):
            return True

        if self._load_error:
            return False

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            self._load_error = f'Model dependencies unavailable: {exc}'
            LOGGER.warning(self._load_error)
            return False

        try:
            required_files = [
                self.encoder_dir / 'config.json',
                self.encoder_dir / 'model.safetensors',
                self.classifier_path,
                self.model_dir / 'tokenizer_config.json',
                self.model_dir / 'vocab.txt',
            ]
            for required_path in required_files:
                if not required_path.exists():
                    self._load_error = f'Missing model artifact: {required_path}'
                    LOGGER.warning(self._load_error)
                    return False

            encoder = self._load_encoder()
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir),
                trust_remote_code=True,
                local_files_only=True,
            )

            classifier_state = torch.load(
                self.classifier_path,
                map_location=self.device,
                weights_only=True,
            )
            classifier = torch.nn.Linear(encoder.config.hidden_size, 2)
            classifier.load_state_dict(classifier_state, strict=True)
            classifier.to(self.device)
            classifier.eval()

            self._encoder = encoder
            self._classifier = classifier
            self._tokenizer = tokenizer
            LOGGER.info('GMO classification model loaded from %s on %s', self.model_dir, self.device)
            return True
        except Exception as exc:
            self._load_error = f'Impossible to load GMO model: {exc}'
            LOGGER.exception(self._load_error)
            return False

    def _predict_with_model(self, sequence: str) -> Dict[str, Any]:
        assert self._encoder is not None
        assert self._classifier is not None
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
            outputs = self._encoder(**tokens)
            pooled = self._masked_mean_pool(
                outputs.last_hidden_state,
                tokens['attention_mask'],
            )
            probs = functional.softmax(self._classifier(pooled), dim=1)

        gmo_prob = float(probs[0, 1].item())
        no_gmo_prob = float(probs[0, 0].item())
        reported_gmo_prob = round(gmo_prob, 4)
        reported_no_gmo_prob = round(no_gmo_prob, 4)
        in_uncertainty_zone = UNCERTAINTY_LOW <= gmo_prob <= UNCERTAINTY_HIGH

        label = 'gmo' if gmo_prob >= DECISION_THRESHOLD else 'no-gmo'
        score = gmo_prob if label == 'gmo' else no_gmo_prob

        return {
            'label': label,
            'score': round(score, 4),
            'probabilities': {
                'gmo': reported_gmo_prob,
                'no_gmo': reported_no_gmo_prob,
            },
            'attention': self._build_attention_segments(
                sequence.upper(),
                reported_gmo_prob,
            ),
            'chunks_processed': 1,
            'in_uncertainty_zone': in_uncertainty_zone,
        }

    def _load_encoder(self) -> Any:
        """Load the custom ESM encoder used to produce the saved classifier features."""
        source_candidates = [
            self.encoder_dir / 'modeling_esm.py',
            self.model_dir / 'modeling_esm.py',
            self.model_dir.parent / 'gmo_model_old' / 'modeling_esm.py',
        ]
        source_path = next((path for path in source_candidates if path.exists()), None)
        if source_path is None:
            raise FileNotFoundError(
                'Custom ESM implementation not found. Expected modeling_esm.py '
                f'next to the encoder or in {self.model_dir.parent / "gmo_model_old"}.'
            )

        module_name = f'_gmo_modeling_esm_{id(self)}'
        module_spec = importlib.util.spec_from_file_location(
            module_name,
            source_path,
            submodule_search_locations=[str(source_path.parent)],
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f'Unable to import custom ESM implementation from {source_path}')

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        encoder = module.EsmModel.from_pretrained(
            str(self.encoder_dir),
            add_pooling_layer=False,
            local_files_only=True,
        )
        encoder.to(self.device)
        encoder.eval()
        return encoder

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp_min(1)

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
            'attention': self._build_attention_segments(sequence, round(gmo_probability, 4)),
            'chunks_processed': 1,
            'in_uncertainty_zone': UNCERTAINTY_LOW <= gmo_probability <= UNCERTAINTY_HIGH,
        }

    def _motif_hits(self, sequence: str) -> int:
        motifs = ('CAAT', 'TATA', 'GCGC', 'CCGG', 'GGCC')
        return sum(sequence.count(motif) for motif in motifs)

    def _build_attention_segments(
        self,
        sequence: str,
        gmo_probability: float,
    ) -> List[Dict[str, Any]]:
        """Explain GMO evidence only when the reported GMO rate is non-zero."""
        if gmo_probability <= 0:
            return []

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

        return sorted(segments, key=lambda item: item['score'], reverse=True)


_INFERENCE_SERVICE: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _INFERENCE_SERVICE
    if _INFERENCE_SERVICE is None:
        _INFERENCE_SERVICE = InferenceService()
    return _INFERENCE_SERVICE
