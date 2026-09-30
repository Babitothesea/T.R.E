"""Trova i pacchetti installati non piu' raggiungibili da nessuna radice.

Dopo aver tolto un pacchetto pesante restano librerie che erano solo sue
dipendenze. Questo script calcola la chiusura delle dipendenze a partire dalle
radici realmente necessarie e stampa cio' che rimarrebbe orfano, senza toccare
nulla: serve a decidere, non a fare.
"""

from __future__ import annotations

import sys
from importlib.metadata import distributions

from packaging.markers import UndefinedEnvironmentName
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# radici: cio' che serve davvero all'applicazione, piu' gli attrezzi del pip
ROOTS = {"opencv-python", "numpy", "onnxruntime", "pip", "setuptools", "wheel"}

installed: dict[str, str] = {}
requires: dict[str, list[Requirement]] = {}
for dist in distributions():
    name = canonicalize_name(dist.metadata["Name"] or "")
    if not name:
        continue
    installed[name] = dist.version
    parsed: list[Requirement] = []
    for raw in dist.requires or []:
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001 - i marker non validi non devono fermarci
            continue
        if req.marker is None:
            parsed.append(req)
            continue
        try:
            if not req.marker.evaluate():
                continue
        except UndefinedEnvironmentName:
            continue  # marker che richiede un extra: non applicabile
        except Exception:  # noqa: BLE001
            continue
        parsed.append(req)
    requires[name] = parsed


def reach(roots: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = [canonicalize_name(r) for r in roots]
    while stack:
        name = stack.pop()
        if name in seen or name not in installed:
            continue
        seen.add(name)
        for req in requires.get(name, []):
            target = canonicalize_name(req.name)
            if target in installed and target not in seen:
                stack.append(target)
    return seen


live = reach(ROOTS)
orphans = sorted(set(installed) - live)

print(f"pacchetti installati : {len(installed)}")
print(f"raggiungibili         : {len(live)}")
print(f"orfani (candidati)    : {len(orphans)}\n")
for name in orphans:
    print(f"  {name}=={installed[name]}")

if "--apply" in sys.argv:
    import subprocess

    print(f"\nrimuovo {len(orphans)} pacchetti...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *orphans], check=False)
    print("fatto")
else:
    print("\n(si applica solo con --apply)")
