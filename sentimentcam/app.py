"""Loop principale: sorgente video -> volti -> emozioni -> overlay a schermo."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from .config import Config, config_from_args, EMOTION_LABELS_IT
from .detector import FaceDetector
from .emotion import EmotionEngine
from .tracker import MultiFaceTracker, valence_from_probs
from .ui import UI

WINDOW = "SentimentCam"


def open_source(cfg: Config):
    """Apre la sorgente. Restituisce ``(sorgente, riproduzione_continua)``."""
    if cfg.is_camera:
        cap = cv2.VideoCapture(int(cfg.source))
        if not cap.isOpened():
            raise RuntimeError(
                f"Impossibile aprire la webcam {cfg.source}. "
                "Verifica che nessun'altra applicazione la stia usando."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        return cap, True

    if cfg.is_image_source:
        image = cv2.imread(cfg.source)
        if image is None:
            raise RuntimeError(f"Immagine non leggibile: {cfg.source}")
        return image, False

    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {cfg.source}")
    return cap, True


class SentimentCam:
    """Applicazione: mantiene i modelli e la logica dei tasti."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        print("[init] Caricamento detector volti (YuNet, OpenCV)...", flush=True)
        self.detector = FaceDetector(
            score_threshold=cfg.det_score_threshold,
            nms_threshold=cfg.det_nms_threshold,
            min_face_px=cfg.min_face_px,
            max_faces=cfg.max_faces,
            weights_dir=cfg.weights_dir,
        )
        print(f"[init] Detector pronto: {self.detector.weights_path}", flush=True)
        self.emotions = EmotionEngine(cfg.emotion_model, cfg.weights_dir, cfg.threads)
        self.tracker = MultiFaceTracker(cfg.iou_threshold, cfg.max_missed)
        self.ui = UI(cfg.panel_width)
        self._default_panel_width = cfg.panel_width
        self.history: deque[float] = deque(maxlen=cfg.history_len)
        self.mirror = cfg.mirror
        self.shots = 0
        self._frames_seen = 0
        self._last_faces: list = []
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    def process(self, frame, budget: int | None = None):
        """Rileva, classifica e restituisce ``(frame, tracks, sentiment_globale)``.

        ``budget`` sovrascrive ``--max-infer-per-frame``: su un'immagine singola
        non c'e' motivo a differire, quindi si classificano tutti i volti subito.
        """
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # tutto resta in BGR: sia YuNet sia il ritaglio allineato lavorano
        # direttamente su array OpenCV, senza conversioni inutili.
        self._frames_seen += 1
        reuse = self.cfg.det_interval > 1 and self._frames_seen % self.cfg.det_interval != 0
        if reuse and self._last_faces:
            faces = self._last_faces
        else:
            faces = self.detector.detect(frame)
            self._last_faces = faces

        tracks = self.tracker.update(faces)
        now = time.monotonic()

        # Budget di inferenza: si aggiornano prima i volti piu' arretrati, cosi'
        # anche con 4+ volti e budget 1 nessuno resta indietro.
        pending = [
            t for t in tracks if t.samples == 0 or now - t.last_emotion_at >= self.cfg.emotion_interval
        ]
        pending.sort(key=lambda t: (0 if t.samples == 0 else 1, -t.stale_seconds))
        for track in pending[: budget or self.cfg.max_infer_per_frame]:
            try:
                # il ritaglio usa la risoluzione nativa del modello
                crop = self.detector.crop(frame, track.face, self.emotions.size)
                probs = self.emotions.classify(crop)
            except Exception as exc:  # un volto non deve fermare gli altri
                print(f"[warn] classificazione fallita per {track.label}: {exc}", flush=True)
                continue
            track.observe(probs, valence_from_probs(probs), self.cfg.smooth, now)

        return frame, tracks, self._overall(tracks)

    @staticmethod
    def _overall(tracks: list) -> float:
        """Sentiment di gruppo: media delle valenze pesata per area del volto."""
        scored = [(t.valence, max(1.0, t.face.area)) for t in tracks if t.samples]
        total = sum(weight for _v, weight in scored)
        if total <= 0:
            return 0.0
        return sum(value * weight for value, weight in scored) / total

    # ------------------------------------------------------------------
    def _screenshot(self, canvas) -> None:
        out_dir = Path(self.cfg.screenshot_dir or "screenshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.shots += 1
        path = out_dir / f"sentiment_{datetime.now():%Y%m%d_%H%M%S}_{self.shots:02d}.png"
        if cv2.imwrite(str(path), canvas):
            print(f"[shot] {path}", flush=True)

    def _handle_key(self, key: int, canvas) -> bool:
        """Gestisce un tasto. Ritorna False per uscire."""
        if key in (27, ord("q")):
            return False
        if key == ord("s"):
            self._screenshot(canvas)
        elif key == ord("r"):
            self.tracker.reset()
            self.history.clear()
            print("[reset] tracking e storico azzerati", flush=True)
        elif key == ord("m"):
            self.mirror = not self.mirror
            print(f"[mirror] {'attivo' if self.mirror else 'disattivo'}", flush=True)
        elif key == ord("h"):
            self.ui.panel_width = 0 if self.ui.panel_width else self._default_panel_width
        elif key in (ord("+"), ord("="), ord("]")):
            self.detector.max_faces = min(32, self.detector.max_faces + 1)
            print(f"[volfi] max {self.detector.max_faces}", flush=True)
        elif key in (ord("-"), ord("_"), ord("[")):
            self.detector.max_faces = max(1, self.detector.max_faces - 1)
            print(f"[volfi] max {self.detector.max_faces}", flush=True)
        return True

    # ------------------------------------------------------------------
    def run(self) -> int:
        cfg = self.cfg
        source, is_stream = open_source(cfg)
        writer = None
        fps = 0.0
        last = started = time.monotonic()
        frames = 0

        try:
            while True:
                if is_stream:
                    ok, frame = source.read()
                    if not ok:
                        break
                else:
                    frame = source.copy()

                frames += 1
                # su un'immagine singola non c'e' motivo a differire la
                # classificazione: si processano tutti i volti in una passata
                frame, tracks, overall = self.process(
                    frame, budget=self.cfg.max_faces if not is_stream else None
                )

                now = time.monotonic()
                dt = now - last
                last = now
                if dt > 0:
                    fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / dt
                self._elapsed = now - started

                # il grafico legge un campione ogni due frame: piu' liscio e
                # copre una finestra temporale piu' lunga
                if frames % 2 == 0 or not is_stream:
                    self.history.append(overall)

                canvas = self.ui.render(
                    frame, tracks, overall, self.history, fps, now - started
                )

                if cfg.save_video:
                    if writer is None:
                        height, width = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            cfg.save_video,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(fps, 10.0),
                            (width, height),
                        )
                    writer.write(canvas)

                if cfg.show_window:
                    if not is_stream:
                        # immagine singola: si mostra e si aspetta la chiusura
                        cv2.imshow(WINDOW, canvas)
                        self._handle_key(cv2.waitKey(0) & 0xFF, canvas)
                        cv2.destroyWindow(WINDOW)
                        break
                    cv2.imshow(WINDOW, canvas)
                    if not self._handle_key(cv2.waitKey(1) & 0xFF, canvas):
                        break
                elif not is_stream:
                    # niente finestra e niente video: un solo frame, poi esce
                    break

                if cfg.max_frames and frames >= cfg.max_frames:
                    break
        finally:
            if writer is not None:
                writer.release()
            if is_stream:
                source.release()
            if cfg.show_window:
                cv2.destroyAllWindows()
            self._summary(frames)

        return 0

    def _summary(self, frames: int) -> None:
        print(f"\n[fine] {frames} frame elaborati", flush=True)
        if frames > 1 and self._elapsed > 0:
            print(f"[fine] media {frames / self._elapsed:.1f} fps", flush=True)
        history = list(self.history)
        if history:
            mean = sum(history) / len(history)
            print(f"[fine] sentiment medio: {mean:+.2f}", flush=True)
        for track in sorted(self.tracker._tracks.values(), key=lambda t: t.track_id):
            if not track.samples:
                continue
            print(
                f"[fine] volto {track.label}: {EMOTION_LABELS_IT[track.dominant]} "
                f"({track.dominant_conf:.0%}) sentiment {track.valence:+.2f} "
                f"inquadrato per {track.hits} frame",
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    cfg = config_from_args(argv)
    try:
        app = SentimentCam(cfg)
    except ImportError as exc:
        print(f"[errore] dipendenza mancante: {exc}", flush=True)
        print("Esegui: pip install -r requirements.txt", flush=True)
        return 2
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
