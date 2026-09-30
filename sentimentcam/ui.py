"""Disegno dell'overlay: box sui volti, pannello laterale e grafico di andamento."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .config import EMOTION_COLORS, EMOTION_LABELS_IT, EMOTIONS

POSITIVE = (110, 210, 120)
NEGATIVE = (70, 70, 235)
NEUTRAL = (175, 175, 175)
BG = (28, 28, 32)
FG = (235, 235, 235)
DIM = (150, 150, 155)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sentiment_color(valence: float) -> tuple[int, int, int]:
    """Colore associato a un punteggio di sentiment in [-1, 1]."""
    if valence > 0.15:
        return POSITIVE
    if valence < -0.15:
        return NEGATIVE
    return NEUTRAL


class UI:
    """Composizione del frame annotato e del pannello informative."""

    def __init__(self, panel_width: int = 300) -> None:
        self.panel_width = panel_width

    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        tracks: list,
        overall: float,
        history: deque,
        fps: float,
        elapsed: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        self._draw_faces(canvas, tracks)
        if self.panel_width <= 0:
            return canvas
        panel = self._draw_panel(canvas.shape[0], tracks, overall, history, fps, elapsed)
        return np.hstack([canvas, panel])

    # ------------------------------------------------------------------
    def _draw_faces(self, canvas: np.ndarray, tracks: list) -> None:
        for track in tracks:
            x1, y1, x2, y2 = (int(round(v)) for v in track.face.xyxy)
            color = sentiment_color(track.valence) if track.samples else DIM

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            if track.samples:
                label = f"{track.label} {EMOTION_LABELS_IT[track.dominant]} {track.dominant_conf:.0%}"
            else:
                label = f"{track.label} ..."
            self._chip(canvas, (x1, y1), label, color)

            # Barra di intensita' del sentiment lungo il bordo inferiore.
            if track.samples and abs(track.valence) > 0.02:
                w = max(1, int((x2 - x1) * min(1.0, abs(track.valence))))
                if track.valence > 0:
                    cv2.rectangle(canvas, (x2 - w, y2 - 6), (x2, y2), color, -1)
                else:
                    cv2.rectangle(canvas, (x1, y2 - 6), (x1 + w, y2), color, -1)

    @staticmethod
    def _chip(canvas: np.ndarray, anchor: tuple[int, int], text: str, color) -> None:
        x, y = anchor
        scale, thick = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
        ty = max(th + 6, y - 6)
        cv2.rectangle(canvas, (x, ty - th - 6), (x + tw + 8, ty + 2), color, -1)
        cv2.putText(canvas, text, (x + 4, ty - 3), FONT, scale, (20, 20, 20), thick, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def _draw_panel(
        self,
        height: int,
        tracks: list,
        overall: float,
        history: deque,
        fps: float,
        elapsed: float,
    ) -> np.ndarray:
        w = self.panel_width
        panel = np.full((height, w, 3), BG, np.uint8)
        cv2.line(panel, (0, 0), (0, height), (60, 60, 65), 1)

        y = 24
        cv2.putText(panel, "SentimentCam", (14, y), FONT, 0.62, FG, 1, cv2.LINE_AA)
        y += 20
        cv2.putText(
            panel,
            f"volfi {len(tracks)}   {fps:.1f} fps   {elapsed:.0f}s",
            (14, y),
            FONT,
            0.42,
            DIM,
            1,
            cv2.LINE_AA,
        )

        y = 66
        cv2.putText(panel, "SENTIMENT DI GRUPPO", (14, y), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        y += 26
        self._gauge(panel, (14, y, w - 28, 10), overall)

        y += 40
        cv2.putText(panel, "ANDAMENTO", (14, y), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        self._sparkline(panel, (14, y + 6, w - 28, 66), history)

        y += 96
        cv2.putText(
            panel, f"VOLTI RILEVATI ({len(tracks)})", (14, y), FONT, 0.42, DIM, 1, cv2.LINE_AA
        )
        y += 12
        for track in tracks:
            if y > height - 96:
                cv2.putText(panel, "...", (14, y), FONT, 0.40, DIM, 1, cv2.LINE_AA)
                break
            y = self._track_row(panel, track, y, w)

        y = height - 74
        for line in ("q esci    s screenshot", "r azzera   m mirror", "h pannello   +/- max volti"):
            cv2.putText(panel, line, (14, y), FONT, 0.38, DIM, 1, cv2.LINE_AA)
            y += 16
        return panel

    @staticmethod
    def _track_row(panel: np.ndarray, track, y: int, w: int) -> int:
        """Una riga per volto: etichetta, confidenza e barra di valenza."""
        x = 14
        right = w - 14
        if track.samples:
            label = track.label
            emotion = EMOTION_LABELS_IT[track.dominant]
            conf = f"{track.dominant_conf:.0%}"
            color = EMOTION_COLORS.get(track.dominant, NEUTRAL)
        else:
            label, emotion, conf, color = track.label, "in analisi...", "", DIM

        cv2.putText(panel, label, (x, y), FONT, 0.42, FG, 1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        cv2.putText(panel, emotion, (x + lw + 10, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        if conf:
            (cw, _), _ = cv2.getTextSize(conf, FONT, 0.42, 1)
            cv2.putText(panel, conf, (right - cw, y), FONT, 0.42, FG, 1, cv2.LINE_AA)

        # barra di valenza: centro = neutro, destra = positivo
        bar_y = y + 6
        cv2.rectangle(panel, (x, bar_y), (right, bar_y + 4), (48, 48, 53), -1)
        mid = (x + right) // 2
        if track.samples:
            value = max(-1.0, min(1.0, track.valence))
            end = int(mid + value * ((right - x) // 2))
            if end >= mid:
                cv2.rectangle(panel, (mid, bar_y), (end, bar_y + 4), sentiment_color(value), -1)
            else:
                cv2.rectangle(panel, (end, bar_y), (mid, bar_y + 4), sentiment_color(value), -1)
        cv2.line(panel, (mid, bar_y - 2), (mid, bar_y + 6), (90, 90, 95), 1)
        return y + 18

    @staticmethod
    def _gauge(panel: np.ndarray, rect: tuple[int, int, int, int], value: float) -> None:
        x, y, w, h = rect
        cy = y + h // 2
        cv2.rectangle(panel, (x, cy - 3), (x + w, cy + 3), (60, 60, 65), -1)
        mid = x + w // 2
        half = w // 2 - 1
        value = max(-1.0, min(1.0, value))
        end = int(mid + value * half)
        if end >= mid:
            cv2.rectangle(panel, (mid, cy - 3), (end, cy + 3), POSITIVE, -1)
        else:
            cv2.rectangle(panel, (end, cy - 3), (mid, cy + 3), NEGATIVE, -1)
        cv2.line(panel, (mid, cy - 8), (mid, cy + 8), DIM, 1)
        cv2.circle(panel, (end, cy), 4, sentiment_color(value), -1)
        # il valore sta sopra la barra: sotto si sovrapporrebbe al riempimento
        text = f"{value:+.2f}"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.42, 1)
        tx = min(x + w - tw, max(x, end + (8 if value >= 0 else -tw - 8)))
        cv2.putText(panel, text, (tx, y - 5), FONT, 0.42, FG, 1, cv2.LINE_AA)
        cv2.putText(panel, "-1", (x, y + h + 11), FONT, 0.34, DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, "+1", (x + w - 12, y + h + 11), FONT, 0.34, DIM, 1, cv2.LINE_AA)

    @staticmethod
    def _sparkline(panel: np.ndarray, rect: tuple[int, int, int, int], history: deque) -> None:
        x, y, w, h = rect
        cv2.rectangle(panel, (x, y), (x + w, y + h), (48, 48, 53), -1)
        mid = y + h // 2
        cv2.line(panel, (x + 2, mid), (x + w - 2, mid), (75, 75, 80), 1)
        if len(history) < 2:
            cv2.putText(panel, "raccolgo dati...", (x + 6, y + 20), FONT, 0.38, DIM, 1, cv2.LINE_AA)
            return

        values = list(history)[-max(2, w):]
        step = (w - 4) / (len(values) - 1)
        points = []
        for i, value in enumerate(values):
            px = int(x + 2 + i * step)
            py = int(mid - max(-1.0, min(1.0, value)) * (h / 2 - 3))
            points.append((px, py))
        for i, ((ax, ay), (bx, by)) in enumerate(zip(points, points[1:])):
            # ogni segmento assume il colore del sentiment che rappresenta.
            mean_value = (values[i] + values[i + 1]) / 2.0
            cv2.line(panel, (ax, ay), (bx, by), sentiment_color(mean_value), 2, cv2.LINE_AA)
        cv2.circle(panel, points[-1], 3, sentiment_color(values[-1]), -1)


__all__ = ["UI", "EMOTION_COLORS", "EMOTIONS", "sentiment_color"]
