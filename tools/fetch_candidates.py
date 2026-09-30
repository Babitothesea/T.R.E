"""Scarica candidati da Wikimedia e li compone in un contact sheet per ispezione.

Le foto scelte vanno viste a occhio: un'espressione che a noi sembra triste
puo' essere neutra per un modello, e viceversa. Serve un set di riferimento
inequivocabile per capire se l'errore e' del modello o del nostro.
"""

from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

WANTED = {
    "cry": "Crying boy portrait tears",
    "rage": "Angry man shouting rage",
    "wow": "Surprised man open mouth surprise",
    "frown": "Frowning man portrait",
    "grin": "Laughing man portrait big smile",
    "worried": "Worried man portrait",
}

# Wikimedia respinge le richieste senza User-Agent.
HEADERS = {"User-Agent": "SentimentCam-dev/1.0 (local validation script)"}


def _open(url: str, timeout: int = 60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout)


def search(query: str, limit: int = 8) -> list[str]:
    """Restituisce i titoli dei file trovati su Commons."""
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote('filetype:bitmap ' + query)}"
        f"&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url&format=json"
    )
    with _open(url) as response:
        import json

        data = json.load(response)
    pages = data.get("query", {}).get("pages", {})
    return sorted(p["title"] for p in pages.values() if p.get("imageinfo"))


def fetch_thumb(title: str, dest: Path, width: int = 640) -> bool:
    """Scarica una versione ridotta: Special:FilePath con ?width= (thumbnail)."""
    name = title.split(":", 1)[-1]
    thumb = (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        f"{urllib.parse.quote(name)}?width={width}"
    )
    try:
        with _open(thumb) as response:
            raw = response.read()
    except Exception as exc:  # noqa: BLE001
        print(f"  download fallito: {exc}")
        return False
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return False
    scale = width / image.shape[1]
    if scale < 1:
        image = cv2.resize(image, (width, int(image.shape[0] * scale)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), image)
    return True


def main() -> int:
    sheet_cells: list[tuple[str, np.ndarray]] = []
    for key, query in WANTED.items():
        print(f"[{key}] {query}")
        got = 0
        for url in search(query):
            dest = Path(f"samples/cand_{key}_{got}.jpg")
            if fetch_thumb(url, dest):
                image = cv2.imread(str(dest))
                cell = cv2.resize(image, (320, 240))
                cv2.putText(cell, f"{key}{got}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                sheet_cells.append((f"{key}{got}", cell))
                got += 1
            time.sleep(0.7)  # Wikimedia non gradisce richieste in raffica
            if got >= 3:
                break

    if not sheet_cells:
        print("nessun candidato scaricato")
        return 1
    cols = 3
    rows = (len(sheet_cells) + cols - 1) // cols
    sheet = np.full((rows * 240, cols * 320, 3), 20, np.uint8)
    for i, (_name, cell) in enumerate(sheet_cells):
        r, c = divmod(i, cols)
        sheet[r * 240 : (r + 1) * 240, c * 320 : (c + 1) * 320] = cell
    cv2.imwrite("samples/_contact_sheet.jpg", sheet)
    print(f"\ncontact sheet: samples/_contact_sheet.jpg ({len(sheet_cells)} candidati)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
