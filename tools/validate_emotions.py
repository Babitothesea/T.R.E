"""Benchmark e validazione end-to-end della pipeline.

Misura i costi reali sulla macchina corrente e verifica la coerenza fra
rilevamento, classificazione e calcolo del sentiment.

Uso:  python tools/validate_emotions.py samples/three.jpg samples/happy.jpg
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentimentcam.config import EMOTION_LABELS_IT, EMOTIONS  # noqa: E402
from sentimentcam.detector import FaceDetector  # noqa: E402
from sentimentcam.emotion import EmotionEngine  # noqa: E402
from sentimentcam.tracker import MultiFaceTracker, valence_from_probs  # noqa: E402


def timeit(fn, repeats: int = 10) -> tuple[float, object]:
    fn()  # warm-up
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    return (time.perf_counter() - start) / repeats * 1000, result


def main(paths: list[str]) -> int:
    detector = FaceDetector()
    engine = EmotionEngine()
    tracker = MultiFaceTracker()

    det_ms: list[float] = []
    emo_ms: list[float] = []
    faces_seen = 0

    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"[salta] {raw} non trovato")
            continue
        image = cv2.imread(str(path))
        height, width = image.shape[:2]

        ms, faces = timeit(lambda: detector.detect(image), repeats=5)
        det_ms.append(ms)
        print(f"\n=== {path.name}  ({width}x{height})  detection {ms:.0f} ms, {len(faces)} volti")

        for i, face in enumerate(faces):
            faces_seen += 1
            crop = detector.crop(image, face, engine.size)
            gray = crop[:, :, 0] if crop.ndim == 3 else crop
            emo, probs = timeit(lambda: engine.classify(crop))
            emo_ms.append(emo)

            valence = valence_from_probs(probs)
            order = np.argsort(probs)[::-1][:3]
            top = "  ".join(f"{EMOTION_LABELS_IT[EMOTIONS[j]]} {probs[j]:.0%}" for j in order)
            print(
                f"   volto #{i + 1}  det={face.score:.2f}  {emo:.0f} ms  "
                f"crop {crop.shape[1]}x{crop.shape[0]}  ->  {top}   sentiment {valence:+.2f}"
            )

            # controlli di coerenza sulla distribuzione
            assert probs.shape == (len(EMOTIONS),), "dimensione attesa errata"
            assert abs(float(probs.sum()) - 1.0) < 1e-3, "le probabilita' non sommano a 1"
            assert int(np.argmax(probs)) == order[0], "argmax incoerente con l'ordinamento"
            assert -1.0 <= valence <= 1.0, "valenza fuori intervallo"

            track = TrackFactory()
            track.observe(probs, valence, 0.55, 0.0)
            assert track.dominant == EMOTIONS[order[0]], "emozione dominante incoerente"

    # percorso completo: detection + tracking + classificazione, come a runtime
    if paths and Path(paths[0]).exists():
        image = cv2.imread(paths[0])
        fresh = MultiFaceTracker()
        start = time.perf_counter()
        for _ in range(10):
            tracks = fresh.update(detector.detect(image))
            for track in tracks:
                track.observe(
                    engine.classify(detector.crop(image, track.face, engine.size)),
                    0.0,
                    0.55,
                    0.0,
                )
        cycle = (time.perf_counter() - start) / 10 * 1000
        print(
            f"\nciclo completo su {len(tracks)} volti: {cycle:.0f} ms/frame "
            f"(~{1000 / cycle:.1f} fps) includendo classify su tutti i volti"
        )

    if det_ms:
        print(f"\ndetection: media {np.mean(det_ms):.0f} ms")
    if emo_ms:
        print(f"emozioni : media {np.mean(emo_ms):.0f} ms per volto, min {np.min(emo_ms):.0f}")
    print(f"volti analizzati in totale: {faces_seen}")
    print("tutti i controlli di coerenza superati")
    return 0


def TrackFactory():
    """Una track vuota, per verificare la coerenza fra probs e Track."""
    from sentimentcam.tracker import Track

    return Track(track_id=1, face=None, first_seen=0.0, last_seen=0.0)


if __name__ == "__main__":
    args = sys.argv[1:] or ["samples/three.jpg", "samples/happy.jpg", "samples/sad.jpg"]
    raise SystemExit(main(args))
