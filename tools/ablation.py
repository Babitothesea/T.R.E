"""Ablation del pre-processing: distingue un errore nostro da un limite del modello.

Se tutte le varianti dicono "neutro" su sad.jpg, il problema non e' il
pre-processing ma il modello (i FER+ over-predicono "neutral" su espressioni
sottili). Se invece una variante trova "sadness", avevamo un bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentimentcam.detector import FaceDetector  # noqa: E402
from sentimentcam.emotion import FER_CLASSES, softmax  # noqa: E402
from sentimentcam.geometry import center_crop  # noqa: E402

sess = ort.InferenceSession("weights/emotion-ferplus-8.onnx", providers=["CPUExecutionProvider"])
FEED = {sess.get_inputs()[0].name: None}
det = FaceDetector()


def predict(gray64: np.ndarray) -> str:
    out = sess.run(None, {sess.get_inputs()[0].name: gray64.astype(np.float32)[None, None]})[0]
    p = softmax(out.ravel())
    order = np.argsort(p)[::-1][:3]
    return "  ".join(f"{FER_CLASSES[i]}:{p[i]:.0%}" for i in order)


VARIANTS = {
    "raw 0-255 (attuale)": lambda g: g,
    "norm (x-127.5)/127.5": lambda g: (g - 127.5) / 127.5,
    "norm x/255": lambda g: g / 255.0,
    "raw 0-1": lambda g: g / 255.0 * 255,
}

for name in ("happy", "sad", "three"):
    image = cv2.imread(f"samples/{name}.jpg")
    faces = det.detect(image)
    for i, face in enumerate(faces):
        aligned = det.crop(image, face, 64)
        plain = center_crop(image, face.xyxy, 64, expand=0.3)
        print(f"\n=== {name} volto #{i + 1}  (conf detector {face.score:.2f})")
        for crop_name, crop in (("allineato", aligned), ("center crop", plain)):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            for vname, fn in VARIANTS.items():
                if vname == "raw 0-1":
                    continue  # identico a raw: non ripetere
                print(f"  {crop_name:<12} {vname:<24} {predict(fn(gray))}")
