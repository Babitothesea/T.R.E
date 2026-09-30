"""Smoke test: YuNet + emotion-ferplus, solo OpenCV e ONNX Runtime.

Verifica che i due grafi carichino, che gli I/O abbiano il formato atteso e che
i tempi siano quelli dichiarati nel README. Confronta anche la versione fp32 con
quella INT8, che su CPU Haswell non porta vantaggi (manca il supporto VNNI).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def align(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = 64) -> np.ndarray:
    src = np.asarray(landmarks, np.float64)
    dst = DST * (size / 112.0)
    src_m, dst_m = src.mean(0), dst.mean(0)
    cov = (dst - dst_m).T @ (src - src_m) / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rot = u @ np.diag(d) @ vt
    scale = (s * d).sum() / (((src - src_m) ** 2).sum() / len(src))
    m = np.zeros((2, 3))
    m[:, :2] = scale * rot
    m[:, 2] = dst_m - scale * (rot @ src_m)
    return cv2.warpAffine(image_bgr, m.astype(np.float32), (size, size), borderValue=0)


def main() -> int:
    det = cv2.FaceDetectorYN.create("weights/librefacerec-det.onnx", "", (320, 320), 0.6, 0.3)
    sessions = {}
    for name in ("8", "12-int8"):
        path = f"weights/emotion-ferplus-{name}.onnx"
        if not Path(path).exists():
            print(f"[salta] {path} assente")
            continue
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
        print(f"### emotion-ferplus-{name}: input {inp.name} {inp.shape} {inp.type} -> {out.name} {out.shape}")
        sessions[name] = sess

    for raw in ("samples/happy.jpg", "samples/sad.jpg", "samples/three.jpg"):
        if not Path(raw).exists():
            continue
        image = cv2.imread(raw)
        h, w = image.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(image)
        faces = [] if faces is None else faces
        print(f"\n=== {raw}: {len(faces)} volti ({w}x{h})")
        for f in faces:
            crop = align(image, f[4:14].reshape(5, 2))
            batch = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[None, None, :, :]
            for name, sess in sessions.items():
                feed = {sess.get_inputs()[0].name: batch}
                sess.run(None, feed)
                t0 = time.perf_counter()
                for _ in range(10):
                    out = sess.run(None, feed)[0]
                ms = (time.perf_counter() - t0) / 10 * 1000
                p = softmax(out.ravel())
                order = np.argsort(p)[::-1][:3]
                top = "  ".join(f"{CLASSES[i]}:{p[i]:.0%}" for i in order)
                print(f"   det={f[-1]:.2f}  {name:>7} {ms:5.1f} ms   {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
