import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from django.conf import settings

LOGGER = logging.getLogger(__name__)
GMO_CONFIDENCE_THRESHOLD = 0.50
MAX_LENGTH = 1024
MODEL_ATTENTION_MAX_SEQUENCE = 1200
CHUNK_SIZE = MAX_LENGTH - 2
CHUNK_STRIDE = CHUNK_SIZE // 2 + CHUNK_SIZE // 4   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceService:
    """Run GMO classification using local model artifacts with graceful fallback."""

    def __init__(self):
        self.model_dir = Path(settings.ML_MODEL_DIR) / 'gmo_model'
        self.device = DEVICE
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._functional: Optional[Any] = None
        self._load_error: Optional[str] = None

    def predict(self, sequence: str) -> Dict[str, Any]:
        if not sequence:
            return {
                'label': 'no-gmo',
                'score': 0.0,
                'probabilities': {'gmo': 0.0, 'no_gmo': 1.0},
                'attention': [],
            }

        if self._ensure_model_loaded():
            return self._predict_with_model(sequence)

        # Keep API responsive when model dependencies/artifacts are unavailable.
        return self._predict_with_fallback(sequence)

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True

        if self._load_error:
            return False

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            self._load_error = f'Model dependencies unavailable: {exc}'
            LOGGER.warning(self._load_error)
            return False

        try:
            config_path = self.model_dir / 'config.pt'
            encoder_path = self.model_dir / 'encoder'
            tokenizer_path = self.model_dir / 'tokenizer'
            classifier_path = self.model_dir / 'classifier.pt'

            if not config_path.exists() or not encoder_path.exists() or not classifier_path.exists() or not tokenizer_path.exists():
                self._load_error = 'One or more model artifact files are missing.'
                LOGGER.warning(self._load_error)
                return False

            config = torch.load(str(config_path), map_location=self.device, weights_only=False)
            hidden_size = int(config['hidden_size'])

            encoder = AutoModel.from_pretrained(
                str(encoder_path),
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation='eager',
            )

            classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )

            classifier_ckpt = torch.load(str(classifier_path), map_location=self.device, weights_only=False)
            classifier.load_state_dict(classifier_ckpt['classifier_state_dict'])

            class _GMOClassifier(nn.Module):
                def __init__(self, internal_encoder: Any, internal_classifier: Any):
                    super().__init__()
                    self.encoder = internal_encoder
                    self.classifier = internal_classifier

                def forward(
                    self,
                    input_ids: Any,
                    attention_mask: Any,
                    compute_attentions: bool,
                ) -> Dict[str, Any]:
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=compute_attentions,
                    )
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                    logits = self.classifier(cls_emb)
                    return {'logits': logits, 'attentions': outputs.attentions if compute_attentions else None}

            model = _GMOClassifier(encoder, classifier)
            model.to(self.device)
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                trust_remote_code=True,
                local_files_only=True,
            )

            self._model = model
            self._tokenizer = tokenizer
            self._torch = torch
            self._functional = functional
            LOGGER.info('GMO model loaded from %s', self.model_dir)
            return True
        except Exception as exc:
            self._load_error = f'Failed to load GMO model: {exc}'
            LOGGER.exception(self._load_error)
            return False

    def _predict_with_model(self, sequence: str) -> Dict[str, Any]:
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None
        assert self._functional is not None

        seq_upper = sequence.upper()
        seq_len = len(seq_upper)

        # Build overlapping chunks when the sequence exceeds one window.
        if seq_len <= CHUNK_SIZE:
            chunk_ranges: List[tuple] = [(0, seq_len)]
        else:
            chunk_ranges = []
            start = 0
            while start < seq_len:
                end = min(start + CHUNK_SIZE, seq_len)
                chunk_ranges.append((start, end))
                if end >= seq_len:
                    break
                start += CHUNK_STRIDE

        gmo_probs: List[float] = []
        no_gmo_probs: List[float] = []
        all_segments: List[Dict[str, Any]] = []

        for chunk_start, chunk_end in chunk_ranges:
            chunk = seq_upper[chunk_start:chunk_end]
            gmo_p, no_gmo_p, segs = self._infer_chunk(chunk, offset=chunk_start, full_seq_len=seq_len)
            gmo_probs.append(gmo_p)
            no_gmo_probs.append(no_gmo_p)
            all_segments.extend(segs)

        avg_gmo = sum(gmo_probs) / len(gmo_probs)
        avg_no_gmo = sum(no_gmo_probs) / len(no_gmo_probs)
        total = avg_gmo + avg_no_gmo
        if total > 0:
            avg_gmo /= total
            avg_no_gmo /= total

        label = 'gmo' if avg_gmo >= GMO_CONFIDENCE_THRESHOLD else 'no-gmo'
        confidence = avg_gmo if label == 'gmo' else avg_no_gmo

        return {
            'label': label,
            'score': round(float(confidence), 4),
            'probabilities': {
                'gmo': round(avg_gmo, 4),
                'no_gmo': round(avg_no_gmo, 4),
            },
            'attention': self._merge_attention_segments(all_segments),
            'chunks_processed': len(chunk_ranges),
        }

    def _infer_chunk(self, chunk: str, offset: int, full_seq_len: int) -> tuple:
        """Tokenize and run inference on a single sequence window."""
        torch = self._torch
        functional = self._functional
        assert torch is not None
        assert functional is not None

        tokens = self._tokenizer(
            chunk,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        compute_attentions = full_seq_len <= MODEL_ATTENTION_MAX_SEQUENCE

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                compute_attentions=compute_attentions,
            )
            probabilities = functional.softmax(outputs['logits'], dim=1)

        gmo_p = float(probabilities[0, 1].item())
        no_gmo_p = float(probabilities[0, 0].item())

        token_scores = self._extract_attention_scores(outputs.get('attentions'), attention_mask)
        segments = self._build_attention_segments(chunk, token_scores, offset=offset)

        return gmo_p, no_gmo_p, segments

    def _merge_attention_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate overlapping segments from multiple chunks, returning the top 6."""
        if not segments:
            return []

        ranked = sorted(segments, key=lambda s: s['score'], reverse=True)
        merged: List[Dict[str, Any]] = []

        for seg in ranked:
            seg_len = max(seg['end'] - seg['start'], 1)
            seg_mid = (seg['start'] + seg['end']) / 2
            overlap = any(
                abs(seg_mid - (k['start'] + k['end']) / 2) < seg_len * 0.6
                for k in merged
            )
            if not overlap:
                merged.append(seg)
            if len(merged) == 6:
                break

        return merged

    def _extract_attention_scores(self, attentions: Any, attention_mask: Any) -> List[float]:
        if not attentions:
            return []

        torch = self._torch
        assert torch is not None

        try:
            # Shape: [layers][batch, heads, tokens, tokens]
            last_layer = attentions[-1][0]  # [heads, tokens, tokens]
            cls_attention = last_layer[:, 0, :]  # [heads, tokens]
            token_scores = cls_attention.mean(dim=0)  # [tokens]
            mask = attention_mask[0].bool().cpu()
            filtered = token_scores.detach().cpu()[mask]
            values = filtered.tolist()
            if values and len(values) > 2:
                # Drop special tokens at both ends when possible.
                values = values[1:-1]
            return [float(v) for v in values]
        except Exception:
            return []

    def _build_attention_segments(self, sequence: str, token_scores: List[float], offset: int = 0) -> List[Dict[str, Any]]:
        if not sequence:
            return []

        sequence_length = len(sequence)
        if sequence_length == 0:
            return []

        def _with_offset(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not offset:
                return segs
            return [{'start': s['start'] + offset, 'end': s['end'] + offset, 'sequence': s['sequence'], 'score': s['score']} for s in segs]

        if not token_scores:
            return _with_offset(self._fallback_attention_segments(sequence))

        ranked_indices = sorted(
            range(len(token_scores)),
            key=lambda idx: token_scores[idx],
            reverse=True,
        )

        segments: List[Dict[str, Any]] = []
        used_centers: List[int] = []
        window = min(24, max(10, sequence_length // 8 or 10))

        for token_idx in ranked_indices:
            normalized = token_idx / max(len(token_scores) - 1, 1)
            center = int(normalized * (sequence_length - 1))

            if any(abs(center - existing) < window // 2 for existing in used_centers):
                continue

            start = max(0, center - window // 2)
            end = min(sequence_length, start + window)
            subsequence = sequence[start:end]
            if not subsequence:
                continue

            score = min(max(float(token_scores[token_idx]), 0.0), 1.0)
            segments.append(
                {
                    'start': start + offset,
                    'end': end + offset,
                    'sequence': subsequence,
                    'score': round(score, 4),
                }
            )
            used_centers.append(center)

            if len(segments) == 6:
                break

        if not segments:
            return _with_offset(self._fallback_attention_segments(sequence))

        return segments

    def _predict_with_fallback(self, sequence: str) -> Dict[str, Any]:
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence)
        motif_hits = self._motif_hits(sequence)

        gmo_probability = min(max(0.18 + (gc_ratio * 0.9) + (motif_hits * 0.05), 0.02), 0.98)
        no_gmo_probability = 1 - gmo_probability
        label = 'gmo' if gmo_probability >= GMO_CONFIDENCE_THRESHOLD else 'no-gmo'
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
