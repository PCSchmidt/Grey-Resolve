"""SCRFD face detector adapter (onnxruntime over det_10g.onnx).

Wraps ``insightface.model_zoo.scrfd.SCRFD`` directly -- deliberately bypassing
``FaceAnalysis`` / ``model_store`` (insightface 2.1's legacy model_store module
is broken; the low-level model-zoo classes work fine). detect() returns
FaceObservation objects sorted by detection score; detect_raw() additionally
exposes the 5-point landmarks needed for alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grey_resolve.types import FaceObservation

DEFAULT_DET_MODEL = Path.home() / ".insightface" / "models" / "det_10g.onnx"


class ScrfdDetector:
    """FaceDetector over the SCRFD det_10g model."""

    def __init__(
        self,
        model_file: str | Path = DEFAULT_DET_MODEL,
        det_thresh: float = 0.5,
        det_size: tuple[int, int] = (640, 640),
    ):
        from insightface.model_zoo.scrfd import SCRFD  # heavy import, deferred

        model_file = Path(model_file)
        if not model_file.exists():
            raise FileNotFoundError(
                f"SCRFD weights not found at {model_file}; run scripts/fetch_backbone.py first"
            )
        self._model = SCRFD(str(model_file))
        self._model.prepare(ctx_id=-1, det_thresh=det_thresh)
        self._det_thresh = det_thresh
        self._det_size = det_size

    def detect_raw(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (det [N,5] bbox+score, kpss [N,5,2] landmarks)."""
        return self._model.detect(image_bgr, input_size=self._det_size, det_thresh=self._det_thresh)

    def detect(self, image: "np.ndarray") -> list[FaceObservation]:
        det, kpss = self.detect_raw(image)
        observations = [
            FaceObservation(
                bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                detection_score=float(row[4]),
                landmarks=None if kpss is None else np.asarray(kpss[i], dtype=np.float32),
            )
            for i, row in enumerate(det)
        ]
        observations.sort(key=lambda o: o.detection_score, reverse=True)
        return observations
