"""Download e verifica dei pesi ONNX.

Tutto lo stack e' composto da file .onnx serviti da URL pubblici: niente
PyTorch, niente ambienti di training. Al primo utilizzo i file mancanti vengono
scaricati in ``weights/`` e controllati con lo SHA-256.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path("weights")

_HF = "https://huggingface.co/onnxmodelzoo"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet"


@dataclass(frozen=True)
class ModelSpec:
    """Un peso scaricabile: nome, URL, checksum atteso e licenza."""

    filename: str
    url: str
    sha256: str | None
    license: str
    size_mb: int
    description: str


MODELS: dict[str, ModelSpec] = {
    # Rilevamento volti: YuNet, dal model zoo ufficiale OpenCV.
    "yunet": ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=f"{_ZOO}/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        license="MIT",
        size_mb=0,
        description="YuNet 2023mar - detector volti con 5 landmark",
    ),
    # Espressioni facciali: ONNX Model Zoo, addestrato su FER+ (8 classi).
    "emotion-ferplus-8": ModelSpec(
        filename="emotion-ferplus-8.onnx",
        url=f"{_HF}/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx",
        sha256=None,
        license="MIT (modello) / Apache-2.0 (card Hugging Face)",
        size_mb=33,
        description="FER+ fp32 -accuratezza massima",
    ),
    "emotion-ferplus-12-int8": ModelSpec(
        filename="emotion-ferplus-12-int8.onnx",
        url=f"{_HF}/emotion-ferplus-12-int8/resolve/main/emotion-ferplus-12-int8.onnx",
        sha256=None,
        license="MIT (modello) / Apache-2.0 (card Hugging Face)",
        size_mb=19,
        description="FER+ quantizzato INT8 - piu' veloce su CPU",
    ),
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            read += len(block)
            if total:
                pct = read * 100 // total
                print(f"\r  scarico {dest.name}: {pct:3d}%", end="", file=sys.stderr, flush=True)
    print("", file=sys.stderr)
    partial.replace(dest)


def ensure_weights(name: str, dest_dir: Path | str = DEFAULT_DIR) -> Path:
    """Restituisce il percorso del peso, scaricandolo se mancante.

    Args:
        name: chiave in :data:`MODELS`, oppure un percorso a un .onnx esistente.
        dest_dir: cartella dove tenere i pesi.
    """
    if name not in MODELS:
        candidate = Path(name)
        if candidate.exists():
            return candidate
        known = ", ".join(sorted(MODELS))
        raise KeyError(f"Modello sconosciuto '{name}'. Disponibili: {known}")

    spec = MODELS[name]
    dest = Path(dest_dir) / spec.filename

    if dest.exists() and spec.sha256 and _sha256(dest) == spec.sha256:
        return dest

    if not dest.exists():
        print(f"[init] Scarico {spec.description} ({spec.size_mb} MB)...", flush=True)
        _download(spec.url, dest)

    if spec.sha256:
        actual = _sha256(dest)
        if actual != spec.sha256:
            raise RuntimeError(
                f"Checksum non corrispondente per {dest}:\n  atteso {spec.sha256}\n  ottenuto {actual}"
            )
    return dest
