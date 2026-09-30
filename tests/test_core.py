"""Test della logica che non dipende dai pesi: tracker, sentiment, UI, config.

Eseguiti senza pytest:  python tests/test_core.py
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentimentcam.config import EMOTIONS, Config  # noqa: E402
from sentimentcam.detector import Face, iou  # noqa: E402
from sentimentcam.tracker import MultiFaceTracker, valence_from_probs  # noqa: E402
from sentimentcam.ui import UI  # noqa: E402

PASSED = 0


def check(condition: bool, label: str) -> None:
    global PASSED
    assert condition, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def make_face(x: float, y: float, size: float = 80.0) -> Face:
    return Face(
        xyxy=(x, y, x + size, y + size),
        score=0.9,
        landmarks=np.array([[x + 30, y + 30], [x + 50, y + 30], [x + 40, y + 45],
                            [x + 32, y + 60], [x + 48, y + 60]], np.float32),
    )


def dist(probs: dict[str, float]) -> np.ndarray:
    return np.array([probs.get(e, 0.0) for e in EMOTIONS], np.float32)


def test_iou() -> None:
    print("iou")
    check(iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0, "box identici -> 1.0")
    check(iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0, "box disgiunti -> 0.0")
    check(abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-6, "sovrapposizione a meta -> 1/3")


def test_valence() -> None:
    print("valence_from_probs")
    check(valence_from_probs(dist({"happy": 1.0})) > 0.99, "felice -> +1")
    check(valence_from_probs(dist({"angry": 1.0})) < -0.85, "arrabbiato -> negativo")
    check(abs(valence_from_probs(dist({"neutral": 1.0}))) < 1e-6, "neutro puro -> 0")
    mixed = valence_from_probs(dist({"happy": 0.5, "neutral": 0.5}))
    check(0.0 < mixed < 0.5, "mezzo mezzo penalizzato dalla massa neutra")
    check(-1.0 <= valence_from_probs(dist({"sad": 0.9, "fear": 0.1})) <= 1.0, "resta in [-1,1]")


def test_tracker_ids_stable() -> None:
    print("MultiFaceTracker: ID stabili su 3 volti")
    tracker = MultiFaceTracker(iou_threshold=0.3, max_missed=3)

    # tre volti affiancati
    t0 = tracker.update([make_face(10, 10), make_face(200, 10), make_face(400, 10)], now=0.0)
    check(len(t0) == 3, "3 volti rilevati")
    check([t.track_id for t in t0] == [1, 2, 3], "ID iniziali 1,2,3")

    # ogni volto si sposta di 5 px: devono mantenere lo stesso ID
    t1 = tracker.update([make_face(15, 12), make_face(205, 11), make_face(403, 14)], now=1.0)
    check([t.track_id for t in t1] == [1, 2, 3], "ID invariati durante il movimento")
    check([t.hits for t in t1] == [2, 2, 2], "hits incrementati")

    # il volto centrale esce dall'inquadratura, poi rientra
    t2 = tracker.update([make_face(20, 14), make_face(405, 16)], now=2.0)
    check([t.track_id for t in t2] == [1, 3], "il volto uscente viene scartato")
    check(tracker.active_count() == 3, "track 2 ancora vivo in attesa")

    t3 = tracker.update([make_face(20, 14), make_face(210, 14), make_face(407, 18)], now=3.0)
    check(2 in [t.track_id for t in t3], "il volto rientra con la sua track")
    check(max(t.track_id for t in t3) == 3, "nessun ID nuovo sprecato")

    # dopo max_missed frame la track sparisce
    for step in range(4, 9):
        tracker.update([make_face(25, 15), make_face(410, 20)], now=float(step))
    check(tracker.active_count() == 2, "track scaduta rimossa dall'elenco")

    tracker.reset()
    check(tracker.active_count() == 0, "reset svuota il tracker")


def test_tracker_emotion_state() -> None:
    print("Track: stato emozionale")
    tracker = MultiFaceTracker()
    track = tracker.update([make_face(0, 0)], now=0.0)[0]
    check(track.samples == 0, "nessuna classificazione iniziale")
    check(track.dominant == "", "senza dati non c'e' etichetta")
    check(track.stale_seconds == float("inf"), "stale infinita prima della prima inferenza")

    track.observe(dist({"happy": 1.0}), 1.0, smooth=0.5, now=0.0)
    check(track.dominant == "happy", "emozione dominante rilevata")
    check(abs(track.valence - 1.0) < 1e-6, "valenza 1.0")

    # 9 osservazioni "triste" con smooth 0.5 -> la EMA converge verso triste
    for i in range(9):
        track.observe(dist({"sad": 1.0}), -0.8, smooth=0.5, now=float(i))
    check(track.dominant == "sad", "la media mobile converge sulla nuova emozione")
    check(track.valence < 0, "la valenza diventa negativa")

    # il round-robin mette per primi i volti mai classificati
    # 4 volti nuovi, lontani dalla track gia' nota
    fresh = [t for t in tracker.update([make_face(200 + i * 150, 0) for i in range(4)], now=10.0)]
    check(len(fresh) == 4, "4 volti nuovi visible")
    check(sum(1 for t in fresh if t.samples == 0) == 4, "4 nuovi volti senza dati")
    check(fresh[0].track_id == 2, "il volto gia' noto conserva l'ID")
    check(fresh[0].stale_seconds == float("inf"), "i nuovi volti hanno priorita' alta")


def test_ui_render() -> None:
    print("UI")
    tracker = MultiFaceTracker()
    tracks = tracker.update([make_face(20, 20), make_face(200, 20), make_face(380, 20)], now=0.0)
    # un'emozione diversa per volto, cosi' il pannello mostra voci eterogenee
    for track, dominant in zip(tracks, ("happy", "sad", "angry")):
        probs = dist({dominant: 1.0})
        track.observe(probs, valence_from_probs(probs), 0.5, 0.0)

    frame = np.zeros((480, 640, 3), np.uint8)
    history = deque([0.1, -0.4, 0.6, -0.2] * 60, maxlen=240)
    ui = UI(panel_width=300)
    canvas = ui.render(frame, tracks, 0.25, history, 24.5, 12.0)
    check(canvas.shape == (480, 940, 3), f"canvas 640+300 px (ottenuto {canvas.shape})")
    check(canvas[:, 640:, :].std() > 0, "il pannello e' stato effettivamente disegnato")

    bare = UI(panel_width=0).render(frame, tracks, 0.25, history, 24.5, 12.0)
    check(bare.shape == (480, 640, 3), "senza pannello il frame resta invariato")

    # sparkline con dati insufficienti non deve rompersi
    short = UI(300).render(frame, [], 0.0, deque([0.1], maxlen=240), 1.0, 0.1)
    check(short.shape == (480, 940, 3), "sparkline con un solo campione")

    # nessun volto
    empty = UI(300).render(frame, [], 0.0, deque(), 1.0, 0.1)
    check(empty.shape == (480, 940, 3), "render con zero volti")

    # tanti volti da far traboccare il pannello: deve tagliare, non rompersi
    many = MultiFaceTracker()
    crowd = many.update([make_face(i * 90, 0) for i in range(12)], now=0.0)
    tall = UI(300).render(np.zeros((360, 640, 3), np.uint8), crowd, 0.0, history, 10.0, 5.0)
    check(tall.shape == (360, 940, 3), "pannello con 12 volti su frame basso")


def test_config() -> None:
    print("Config")
    cfg = Config(max_faces=1)
    cfg.validate()
    check(cfg.max_faces == 3, "max-faces viene alzato al minimo di 3")
    check(Config().max_faces >= 3, "il default soddisfa il requisito 3+ volti")

    try:
        Config(smooth=0.0).validate()
        raise AssertionError("FAIL: smooth=0 doveva essere rifiutato")
    except ValueError:
        check(True, "smooth fuori range rifiutato")

    try:
        Config(det_interval=0).validate()
        raise AssertionError("FAIL: det-interval=0 doveva essere rifiutato")
    except ValueError:
        check(True, "det-interval fuori range rifiutato")

    check(Config(source="0").is_camera, "sorgente numerica = webcam")
    check(Config(source="clip.mp4").is_camera is False, "sorgente file = non webcam")
    check(Config(source="a.jpg").is_image_source, "estensione immagine riconosciuta")


def test_geometry() -> None:
    print("geometry")
    from sentimentcam.geometry import ARCFACE_TEMPLATE_112, align_face, estimate_norm

    size = 64
    template = ARCFACE_TEMPLATE_112 * (size / 112.0)
    m = estimate_norm(template, size)
    check(np.allclose(m[:, :2], np.eye(2), atol=1e-6), "stima dell'identita' sul template")
    check(np.allclose(m[:, 2], 0.0, atol=1e-6), "nessuna traslazione sul template")

    image = np.zeros((200, 200, 3), np.uint8)
    cv2.circle(image, (100, 100), 60, (255, 255, 255), -1)
    box = (40.0, 40.0, 160.0, 160.0)

    # landmark realistici dentro il box, nell'ordine del template ArcFace
    x1, y1, w_box, h_box = box[0], box[1], box[2] - box[0], box[3] - box[1]
    landmarks = np.array(
        [
            [x1 + 0.35 * w_box, y1 + 0.38 * h_box],
            [x1 + 0.65 * w_box, y1 + 0.38 * h_box],
            [x1 + 0.50 * w_box, y1 + 0.58 * h_box],
            [x1 + 0.42 * w_box, y1 + 0.75 * h_box],
            [x1 + 0.58 * w_box, y1 + 0.75 * h_box],
        ],
        np.float64,
    )

    crop = align_face(image, box, landmarks, size)
    check(crop.shape == (size, size, 3), f"ritaglio {size}x{size} (ottenuto {crop.shape})")
    check(crop.mean() > 60, "il ritaglio allineato contiene il volto")

    plain = align_face(image, box, None, size)
    check(plain.shape == (size, size, 3), "ritaglio di fallback senza landmark")
    check(
        abs(float(plain.mean()) - float(crop.mean())) < 20,
        "ritaglio allineato e fallback coprono la stessa regione",
    )


BANNED = ("torch", "libreyolo", "ultralytics", "tensorflow", "mediapipe")


def test_no_heavy_dependencies() -> None:
    """La pipeline deve funzionare senza PyTorch e senza librerie YOLO.

    Bloccando i moduli a livello di import si dimostra che il codice non li
    tocca: se domani un'import dipendente da uno di questi, il test fallisce
    qui invece che in produzione.
    """
    print(f"nessuna dipendenza pesante ({', '.join(BANNED)})")

    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in BANNED:
            raise AssertionError(f"tentato import di '{root}', che non deve servire")
        return real_import(name, *args, **kwargs)

    modules = ("sentimentcam.app", "sentimentcam.detector", "sentimentcam.emotion")
    for module in modules:
        sys.modules.pop(module, None)
    builtins.__import__ = guarded
    try:
        for module in modules:
            importlib.import_module(module)
        check(True, "app/detector/emotion importano senza dipendenze pesanti")
    finally:
        builtins.__import__ = real_import

    check(
        "torch" not in sys.modules and "libreyolo" not in sys.modules,
        "torch e libreyolo restano fuori da sys.modules",
    )


def main() -> int:
    for test in (
        test_iou,
        test_valence,
        test_tracker_ids_stable,
        test_tracker_emotion_state,
        test_ui_render,
        test_geometry,
        test_config,
        test_no_heavy_dependencies,
    ):
        test()
    print(f"\n{PASSED} verifiche superate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
