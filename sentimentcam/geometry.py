"""Geometria del volto: similarity transform e ritaglio allineato.

YuNet restituisce 5 landmark (occhi, naso, angoli della bocca) nell'ordine
convenzionale ArcFace. Da questi si stima una similarity transform
(rotazione + scala uniforme + traslazione, senza shear) che porta il volto su
un template canonico: il ritaglio risultante e' frontale e stabile anche a
testa ruotata, che e' esattamente cio' che serve a un classificatore di
espressioni addestrato su volti frontali.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

#: Template ArcFace a 112x112: [occhio dx, occhio sx, naso, bocca dx, bocca sx].
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float64,
)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Stima a minimi quadrati la similarity transform da ``src`` a ``dst``.

    Ritorna una matrice affine ``(2, 3)`` tale che ``dst ~= src @ M[:, :2].T + M[:, 2]``.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    cov = (dst - dst_mean).T @ src_demean / len(src)
    u, s, vt = np.linalg.svd(cov)

    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var_src = (src_demean**2).sum() / len(src)
    scale = (s * d).sum() / var_src if var_src > 0 else 1.0

    m = np.zeros((2, 3), np.float64)
    m[:, :2] = scale * rot
    m[:, 2] = dst_mean - scale * (rot @ src_mean)
    return m


def estimate_norm(landmarks5: np.ndarray, size: int = 64) -> np.ndarray:
    """Affine ``(2, 3)`` che porta i 5 landmark sul template di lato ``size``."""
    template = ARCFACE_TEMPLATE_112 * (size / 112.0)
    return umeyama_similarity(np.asarray(landmarks5, np.float64), template)


def center_crop(
    image: np.ndarray, box: Sequence[float], size: int, expand: float = 0.0
) -> np.ndarray:
    """Ritaglio quadrato centrato sul box, con espansione opzionale."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) / 2.0 * (1.0 + expand)
    x1 = int(max(0, round(cx - half)))
    y1 = int(max(0, round(cy - half)))
    x2 = int(min(width, round(cx + half)))
    y2 = int(min(height, round(cy + half)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box degenere: {tuple(box)}")
    return cv2.resize(image[y1:y2, x1:x2], (size, size))


def align_face(
    image: np.ndarray,
    box: Sequence[float],
    landmarks: np.ndarray | None = None,
    size: int = 64,
) -> np.ndarray:
    """Ritaglio allineato del volto, ``size`` x ``size``.

    Usa la similarity transform quando i 5 landmark sono disponibili, altrimenti
    ripiega su un ritaglio quadrato centrato sul box.
    """
    if landmarks is not None:
        lm = np.asarray(landmarks, np.float64)
        if lm.shape == (5, 2):
            m = estimate_norm(lm, size).astype(np.float32)
            return cv2.warpAffine(image, m, (size, size), borderValue=0)
    return center_crop(image, box, size)
