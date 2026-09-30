"""Classificazione delle espressioni facciali con ONNX Runtime.

Il modello e' ``emotion-ferplus-8`` dell'ONNX Model Zoo: una CNN addestrata su
FER+ (le annotazioni crowdsourced di FER2013, paper arXiv:1608.01041). Accetta
un'immagine in scala di grigi di 64x64 e restituisce 8 punteggi.

La scelta di FER+ e' il punto: e' addestrato esattamente su espressioni
facciali, quindi distingue un viso triste da uno neutro. Un classificatore
generico non addestrato su volti non ci arriva, e su questa macchina costerebbe
un intero framework di deep learning per farlo.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .weights import DEFAULT_DIR, ensure_weights

#: Le 8 classi di FER+, nell'ordine di uscita del modello.
FER_CLASSES: tuple[str, ...] = (
    "neutral",
    "happy",
    "surprise",
    "sad",
    "angry",
    "disgust",
    "fear",
    "contempt",
)

#: Risoluzione nativa attesa dai modelli FER+ della zoo.
NATIVE_SIZE = 64


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class EmotionEngine:
    """Classificatore di espressioni a partire da ritagli di volto BGR."""

    def __init__(
        self,
        model: str = "emotion-ferplus-8",
        weights_dir: str = str(DEFAULT_DIR),
        threads: int = 0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
            raise ImportError(
                "Il backend emozioni richiede onnxruntime: pip install onnxruntime"
            ) from exc

        path = ensure_weights(model, weights_dir)
        options = ort.SessionOptions()
        # 0 lascia decidere a onnxruntime; con i 2 core fisici di un portatile
        # vecchio over-subscription peggiora i tempi.
        options.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) // 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.model_path = str(path)
        self.labels = list(FER_CLASSES)

        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        shape = list(spec.shape)
        # attende (1, 1, H, W) in scala di grigi
        self.size = int(shape[-1]) if isinstance(shape[-1], int) and shape[-1] > 0 else NATIVE_SIZE
        self.channels = int(shape[-3]) if isinstance(shape[-3], int) else 1
        if len(self.labels) != self.session.get_outputs()[0].shape[-1]:
            raise RuntimeError(
                f"{os.path.basename(path)} produce {self.session.get_outputs()[0].shape[-1]} "
                f"classi, ma FER_CLASSES ne contiene {len(self.labels)}"
            )
        print(
            f"[init] Emozioni: {os.path.basename(path)} "
            f"({os.path.getsize(path) / 1e6:.0f} MB, {self.size}x{self.size}, "
            f"{len(self.labels)} classi) pronto in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    def classify(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """Classifica un ritaglio di volto.

        Args:
            face_crop_bgr: array BGR uint8, quadrato, gia' allineato.

        Returns:
            Vettore float32 di lunghezza ``len(FER_CLASSES)``, normalizzato a 1.
        """
        import cv2

        gray = (
            face_crop_bgr
            if face_crop_bgr.ndim == 2
            else cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        )
        if gray.shape[0] != self.size:
            gray = cv2.resize(
                gray, (self.size, self.size), interpolation=cv2.INTER_AREA
            )
        # FER+ e' addestrato su valori 0-255 in scala di grigi, senza normalizzare
        batch = gray.astype(np.float32).reshape(1, self.channels, self.size, self.size)

        logits = self.session.run(None, {self.input_name: batch})[0].ravel()
        return softmax(logits.astype(np.float64)).astype(np.float32)
