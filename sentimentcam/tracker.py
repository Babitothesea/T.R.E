"""Tracking multi-volto.

Associa i volti rilevati nel frame corrente alle track del frame precedente con
un greedy matching per IoU, cosi' ogni volto mantiene un ID stabile (e quindi
una timeline del sentiment) mentre si muove.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from .config import EMOTIONS, EMOTION_VALENCE
from .detector import Face, iou


@dataclass
class Track:
    """Stato persistente di un volto attraverso i frame."""

    track_id: int
    face: Face
    first_seen: float
    last_seen: float
    last_emotion_at: float = 0.0
    hits: int = 1
    missed: int = 0
    # Distribuzione emozioni smorzata esponenzialmente, indicizzata da EMOTIONS.
    probs: np.ndarray = field(default_factory=lambda: np.zeros(len(EMOTIONS), np.float32))
    # Valenza sentimentale smorzata, in [-1, 1].
    valence: float = 0.0
    samples: int = 0
    # Secondi trascorsi dall'ultima classificazione (inf se mai avvenuta).
    stale_seconds: float = float("inf")

    @property
    def dominant(self) -> str:
        """Nome dell'emozione dominante."""
        if self.samples == 0:
            return ""
        return EMOTIONS[int(np.argmax(self.probs))]

    @property
    def dominant_conf(self) -> float:
        return float(self.probs.max()) if self.samples else 0.0

    @property
    def label(self) -> str:
        return f"#{self.track_id}"

    def observe(self, probs: np.ndarray, valence: float, smooth: float, now: float) -> None:
        """Integra una nuova osservazione emozionale nello stato."""
        if self.samples == 0:
            self.probs = probs.astype(np.float32).copy()
            self.valence = valence
            self.samples = 1
        else:
            self.probs = (1.0 - smooth) * self.probs + smooth * probs.astype(np.float32)
            self.valence = (1.0 - smooth) * self.valence + smooth * valence
            self.samples += 1
        self.last_emotion_at = now

    def mark_stale(self, now: float) -> None:
        """Aggiorna da quanti secondi questa track e' senza una classificazione."""
        self.stale_seconds = float("inf") if self.samples == 0 else now - self.last_emotion_at


class MultiFaceTracker:
    """Greedy IoU tracker. Nessuna dipendenza esterna oltre NumPy."""

    def __init__(self, iou_threshold: float = 0.30, max_missed: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, faces: list[Face], now: float | None = None) -> list[Track]:
        """Associa i volti del frame alle track e restituisce quelle visibili."""
        now = time.monotonic() if now is None else now

        unmatched_tracks = list(self._tracks.values())
        assignments: list[tuple[Track, Face]] = []

        # greedy: coppia con IoU massimo, una alla volta.
        pairs = []
        for track in unmatched_tracks:
            for idx, face in enumerate(faces):
                score = iou(track.face.xyxy, face.xyxy)
                if score >= self.iou_threshold:
                    pairs.append((score, track.track_id, idx))
        pairs.sort(reverse=True)

        used_tracks: set[int] = set()
        used_faces: set[int] = set()
        for _score, track_id, idx in pairs:
            if track_id in used_tracks or idx in used_faces:
                continue
            track = self._tracks[track_id]
            track.face = faces[idx]
            track.last_seen = now
            track.missed = 0
            track.hits += 1
            track.mark_stale(now)
            assignments.append((track, faces[idx]))
            used_tracks.add(track_id)
            used_faces.add(idx)

        # Track non abbinati: persi per qualche frame ma ancora vivi.
        for track in unmatched_tracks:
            if track.track_id not in used_tracks:
                track.missed += 1
                track.mark_stale(now)

        # Volti nuovi.
        for idx, face in enumerate(faces):
            if idx in used_faces:
                continue
            track = Track(
                track_id=next(self._ids),
                face=face,
                first_seen=now,
                last_seen=now,
            )
            track.mark_stale(now)
            self._tracks[track.track_id] = track
            assignments.append((track, face))

        # Track spariti da troppo tempo.
        for track_id in [t.track_id for t in self._tracks.values() if t.missed > self.max_missed]:
            del self._tracks[track_id]

        visible = [t for t in assignments if t[0].missed == 0]
        visible.sort(key=lambda tf: tf[0].track_id)
        for track, _face in visible:
            track.mark_stale(now)
        return [t for t, _f in visible]

    def reset(self) -> None:
        self._tracks.clear()
        self._ids = itertools.count(1)

    def active_count(self) -> int:
        return len(self._tracks)


def valence_from_probs(probs: np.ndarray) -> float:
    """Converte una distribuzione emozionale in un punteggio [-1, 1].

    Il contributo delle emozioni neutre viene smorzato: un volto perfettamente
    neutro non e' "positivo", e' semplicemente neutro.
    """
    weights = np.array([EMOTION_VALENCE[e] for e in EMOTIONS], np.float32)
    raw = float(np.dot(probs, weights))
    neutral_mass = float(probs[EMOTIONS.index("neutral")])
    intensity = 1.0 - neutral_mass
    return float(np.clip(raw * intensity, -1.0, 1.0))
