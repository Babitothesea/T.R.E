"""SentimentCam - analisi del sentiment da webcam, multi-volto.

Stack deliberatamente minimale: OpenCV (YuNet per il rilevamento dei volti) e
ONNX Runtime (emotion-ferplus per le espressioni facciali). Nessun PyTorch,
nessuna libreria di detection: i due pesi sono file .onnx da 33 MB e 230 KB.
"""

from .config import EMOTIONS, Config

__all__ = ["Config", "EMOTIONS"]
