"""InsightFace ArcFace embedding extractor (primary backbone).

Wraps the w600k_r50 ArcFace ONNX model: 512-d unit-norm embeddings from 112x112
aligned crops. Uses ``insightface.model_zoo.arcface_onnx.ArcFaceONNX`` directly
(bypassing FaceAnalysis / the broken legacy model_store in insightface 2.1).

Weights are downloaded at runtime via scripts/fetch_backbone.py and never
committed (non-commercial grant; see docs/BACKBONE_LICENSES.md).

Preprocessing contract: aligned 112x112 crops produced by
``norm_crop(image, landmarks, 112)`` from the paired SCRFD detector's 5-point
landmarks. The bbox-crop fallback in extract() is for degraded/legacy inputs
only and is measurably worse -- prefer extract_from_landmarks().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grey_resolve.types import EMBEDDING_DIM, FaceObservation, normalize_embedding

MODEL_ID = "insightface/w600k_r50"
DEFAULT_REC_MODEL = Path.home() / ".insightface" / "models" / "w600k_r50.onnx"
ALIGNED_SIZE = 112


class InsightFaceArcFaceExtractor:
    """EmbeddingExtractor over the w600k_r50 ArcFace ONNX model."""

    def __init__(self, model_file: str | Path = DEFAULT_REC_MODEL):
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX  # heavy import, deferred

        model_file = Path(model_file)
        if not model_file.exists():
            raise FileNotFoundError(
                f"ArcFace weights not found at {model_file}; run scripts/fetch_backbone.py first"
            )
        self._model = ArcFaceONNX(str(model_file))
        self._model.prepare(ctx_id=-1)

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def extract_aligned(self, aligned_crop_bgr: np.ndarray) -> np.ndarray:
        """Embed one already-aligned 112x112 BGR crop to a unit-norm 512-d vector."""
        feats = self._model.get_feat([aligned_crop_bgr])
        if feats is None or len(feats) == 0:
            raise ValueError("recognition model returned no embedding")
        return normalize_embedding(np.asarray(feats[0], dtype=np.float32))

    def extract_from_landmarks(self, image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align via the 5-point landmarks (preferred path), then embed."""
        from insightface.utils.face_align import norm_crop

        aligned = norm_crop(image_bgr, landmark=np.asarray(landmarks), image_size=ALIGNED_SIZE)
        return self.extract_aligned(aligned)

    def extract(self, image: "np.ndarray", observation: FaceObservation) -> np.ndarray:
        """Fallback path: bbox crop + resize to 112x112, then embed. Prefer landmarks."""
        import cv2

        x1, y1, x2, y2 = (int(round(v)) for v in observation.bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate bbox {observation.bbox} for image {image.shape}")
        crop = image[y1:y2, x1:x2]
        aligned = cv2.resize(crop, (ALIGNED_SIZE, ALIGNED_SIZE), interpolation=cv2.INTER_LINEAR)
        return self.extract_aligned(aligned)
