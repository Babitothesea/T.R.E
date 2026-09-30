"""Salva i ritagli allineati per capire cosa vede davvero il classificatore."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentimentcam.detector import FaceDetector
from sentimentcam.geometry import center_crop

det = FaceDetector()
for name in ("happy", "sad", "three"):
    path = f"samples/{name}.jpg"
    image = cv2.imread(path)
    faces = det.detect(image)
    marked = image.copy()
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = (int(v) for v in face.xyxy)
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 3)
        for lm in face.landmarks:
            cv2.circle(marked, (int(lm[0]), int(lm[1])), 4, (0, 0, 255), -1)
        # allineato tramite landmark, ingrandito per ispezione
        crop = det.crop(image, face, 224)
        cv2.imwrite(f"samples/_crop_{name}_{i}_aligned.png", crop)
        # ritaglio centrato sul box, senza landmark
        plain = center_crop(image, face.xyxy, 224, expand=0.3)
        cv2.imwrite(f"samples/_crop_{name}_{i}_plain.png", plain)
    cv2.imwrite(f"samples/_marked_{name}.jpg", marked)
    print(f"{name}: {len(faces)} volti, salvati i ritagli")
