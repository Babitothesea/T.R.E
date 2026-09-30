"""Configurazione dell'applicazione e parsing degli argomenti CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

#: Le 8 classi di FER+, nell'ordine di uscita di emotion-ferplus-*.onnx.
EMOTIONS: tuple[str, ...] = (
    "neutral",
    "happy",
    "surprise",
    "sad",
    "angry",
    "disgust",
    "fear",
    "contempt",
)

#: Etichette mostrate a schermo.
EMOTION_LABELS_IT: dict[str, str] = {
    "neutral": "neutro",
    "happy": "felice",
    "surprise": "sorpreso",
    "sad": "triste",
    "angry": "arrabbiato",
    "disgust": "disgustato",
    "fear": "spaventato",
    "contempt": "disdegno",
}

#: Colori BGR per emozione.
EMOTION_COLORS: dict[str, tuple[int, int, int]] = {
    "neutral": (175, 175, 175),
    "happy": (110, 210, 120),
    "surprise": (70, 200, 240),
    "sad": (215, 160, 85),
    "angry": (70, 70, 235),
    "disgust": (120, 190, 95),
    "fear": (205, 120, 225),
    "contempt": (190, 190, 90),
}

#: Valenza sentimentale associata a ogni emozione, in [-1, 1].
EMOTION_VALENCE: dict[str, float] = {
    "happy": 1.00,
    "surprise": 0.30,
    "neutral": 0.00,
    "contempt": -0.40,
    "fear": -0.70,
    "sad": -0.80,
    "disgust": -0.80,
    "angry": -0.90,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Config:
    """Tutti i parametri regolabili dell'applicazione."""

    # Sorgente: indice webcam ("0"), percorso video, percorso immagine.
    source: str = "0"
    # 960x540 e' un compromesso: la detection YuNet costa circa 130 ms a questa
    # risoluzione su CPU lenta, contro i ~280 ms del 720p. Su hardware piu'
    # recente si puo' tornare a 1280x720.
    width: int = 960
    height: int = 540

    # Limite di volti contemporaneamente analizzati. Il requisito e' >= 3.
    max_faces: int = 6

    # Rilevamento volti (YuNet).
    det_score_threshold: float = 0.60
    det_nms_threshold: float = 0.30
    min_face_px: int = 48
    # Rileva i volti ogni N frame e riusa i box nel mezzo. Su CPU lente guadagna
    # fluidita' a prezzo di box leggermente obsoleti, impercettibili a 4-10 fps.
    det_interval: int = 1

    # Classificazione emozioni (emotion-ferplus, ONNX Runtime).
    emotion_model: str = "emotion-ferplus-8"
    weights_dir: str = "weights"
    threads: int = 0  # 0 = scelta automatica
    # Quanti volti classificare per frame: 1 = round-robin, quindi ogni volto
    # viene aggiornato ogni N frame. Alzarlo riduce la latenza percepita.
    max_infer_per_frame: int = 2
    # Coefficiente della media mobile esponenziale sulle probabilita' (0..1).
    smooth: float = 0.55
    # Un volto viene riclassificato solo se piu' vecchio di questo (secondi).
    emotion_interval: float = 0.10

    # Tracking.
    iou_threshold: float = 0.30
    max_missed: int = 15

    # Presentazione.
    mirror: bool = True
    show_window: bool = True
    panel_width: int = 320
    history_len: int = 240

    # Uscite opzionali.
    save_video: str | None = None
    screenshot_dir: str | None = None
    max_frames: int = 0  # 0 = senza limite

    # ------------------------------------------------------------------
    @property
    def is_image_source(self) -> bool:
        return Path(self.source).suffix.lower() in IMAGE_SUFFIXES

    @property
    def is_camera(self) -> bool:
        return self.source.isdigit()

    def validate(self) -> None:
        if self.max_faces < 1:
            raise ValueError("--max-faces deve essere >= 1")
        if self.max_faces < 3:
            print(
                f"[avviso] --max-faces={self.max_faces}: la richiesta e' di almeno "
                "3 volti, aumento il limite.",
                flush=True,
            )
            self.max_faces = 3
        if not 0.0 < self.smooth <= 1.0:
            raise ValueError("--smooth deve essere in (0, 1]")
        if self.max_infer_per_frame < 1:
            raise ValueError("--max-infer-per-frame deve essere >= 1")
        if self.det_interval < 1:
            raise ValueError("--det-interval deve essere >= 1")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Sentiment analysis da webcam, multi-volto (OpenCV + ONNX Runtime).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0", help="indice webcam, video o immagine")
    p.add_argument(
        "--width", type=int, default=960, help="larghezza richiesta alla webcam (meno = piu' fps)"
    )
    p.add_argument("--height", type=int, default=540, help="altezza richiesta alla webcam")
    p.add_argument(
        "--max-faces", type=int, default=6, help="numero massimo di volti analizzati insieme"
    )
    p.add_argument(
        "--det-conf", type=float, default=0.60, help="soglia di confidenza del detector volti"
    )
    p.add_argument("--det-nms", type=float, default=0.30, help="soglia NMS del detector volti")
    p.add_argument("--min-face-px", type=int, default=48, help="dimensione minima volto in px")
    p.add_argument(
        "--det-interval",
        type=int,
        default=1,
        help="rileva i volti ogni N frame, riusando i box nel mezzo (2 = meta fps)",
    )
    p.add_argument(
        "--emotion-model",
        default="emotion-ferplus-8",
        help="modello emozioni: emotion-ferplus-8 (fp32) o emotion-ferplus-12-int8",
    )
    p.add_argument("--weights-dir", default="weights", help="cartella dei pesi .onnx")
    p.add_argument(
        "--threads", type=int, default=0, help="thread ONNX Runtime (0 = scelta automatica)"
    )
    p.add_argument(
        "--max-infer-per-frame",
        type=int,
        default=2,
        help="quanti volti classificare per frame (1 = round-robin)",
    )
    p.add_argument("--smooth", type=float, default=0.55, help="smorzamento temporale 0..1")
    p.add_argument("--emotion-interval", type=float, default=0.10, help="intervallo minimo (s)")
    p.add_argument("--no-mirror", action="store_true", help="disattiva il mirror della webcam")
    p.add_argument("--no-window", action="store_true", help="non aprire la finestra (test/CI)")
    p.add_argument("--no-panel", action="store_true", help="nasconde il pannello laterale")
    p.add_argument("--history-len", type=int, default=240, help="campioni nel grafico")
    p.add_argument("--save-video", default=None, help="salva il video annotato su file")
    p.add_argument("--screenshot-dir", default=None, help="cartella degli screenshot (tasto s)")
    p.add_argument("--max-frames", type=int, default=0, help="ferma dopo N frame (0 = infinito)")
    return p


def config_from_args(argv: list[str] | None = None) -> Config:
    args = build_parser().parse_args(argv)
    cfg = Config(
        source=args.source,
        width=args.width,
        height=args.height,
        max_faces=args.max_faces,
        det_score_threshold=args.det_conf,
        det_nms_threshold=args.det_nms,
        min_face_px=args.min_face_px,
        det_interval=args.det_interval,
        emotion_model=args.emotion_model,
        weights_dir=args.weights_dir,
        threads=args.threads,
        max_infer_per_frame=args.max_infer_per_frame,
        smooth=args.smooth,
        emotion_interval=args.emotion_interval,
        mirror=not args.no_mirror,
        show_window=not args.no_window,
        panel_width=0 if args.no_panel else 320,
        history_len=args.history_len,
        save_video=args.save_video,
        screenshot_dir=args.screenshot_dir,
        max_frames=args.max_frames,
    )
    cfg.validate()
    return cfg
