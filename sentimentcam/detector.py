"""Rilevamento multi-volto con YuNet (OpenCV Zoo), via l'interfaccia di OpenCV.

Non serve alcun framework di deep learning: ``cv2.FaceDetectorYN`` esegue il
grafo ONNX con il runtime integrato in OpenCV. Restituisce un box e 5 landmark
per ogni volto trovato, senza limiti sul numero di volti: 3, 4, 10 sono
tutti casi normali.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .geometry import align_face
from .weights import DEFAULT_DIR, ensure_weights


@dataclass
class Face:
    """Un volto rilevato in un frame."""

    xyxy: tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray | None  # (5, 2) float32

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over Union tra due box in formato xyxy."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    """Wrapper sottile attorno a ``cv2.FaceDetectorYN`` (YuNet)."""

    def __init__(
        self,
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        min_face_px: int = 48,
        max_faces: int = 6,
        weights_dir: str = str(DEFAULT_DIR),
    ) -> None:
        self.weights_path = ensure_weights("yunet", weights_dir)
        self._impl = cv2.FaceDetectorYN.create(
            str(self.weights_path), "", (320, 320), score_threshold, nms_threshold
        )
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        # pubblici e modificabili a runtime (tasti +/-)
        self.min_face_px = min_face_px
        self.max_faces = max_faces

    def detect(self, image_bgr: np.ndarray) -> List[Face]:
        """Rileva i volti in un frame BGR.

        Restituisce al massimo ``max_faces`` volti, dal piu' grande al piu'
        piccolo, scartando quelli sotto ``min_face_px``.
        """
        height, width = image_bgr.shape[:2]
        self._impl.setInputSize((width, height))
        _, detections = self._impl.detect(image_bgr)
        if detections is None:
            return []

        found: List[Face] = []
        for row in detections:
            x, y, w, h = (float(v) for v in row[:4])
            face = Face(
                xyxy=(x, y, x + w, y + h),
                score=float(row[-1]),
                landmarks=row[4:14].reshape(5, 2).astype(np.float32),
            )
            if min(face.width, face.height) < self.min_face_px:
                continue
            found.append(face)

        found.sort(key=lambda f: f.area, reverse=True)
        return found[: self.max_faces]

    def crop(self, image_bgr: np.ndarray, face: Face, size: int = 64) -> np.ndarray:
        """Ritaglio allineato del volto (BGR, ``size`` x ``size``)."""
        return align_face(image_bgr, face.xyxy, face.landmarks, size)
