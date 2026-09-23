"""Shared helpers for the training scripts."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

TRAINING_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = TRAINING_DIR / "config" / "signs.yaml"
RAW_DIR = TRAINING_DIR / "data" / "raw"

SIGN_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Clip filenames encode where the clip came from and who signed it:
#   <source>_<signer>_<clip_id>.mp4   e.g. self_s01_20260923-101500-03.mp4
# The signer ID is what lets step 1.5 split train/val/test BY SIGNER.
CLIP_NAME_RE = re.compile(r"^(?P<source>[a-z]+)_(?P<signer>[a-z0-9]+)_(?P<clip>[A-Za-z0-9-]+)\.mp4$")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for s in cfg["signs"]:
        # YAML 1.1 turns bare yes/no/on/off into booleans; demand real strings.
        if not isinstance(s["name"], str):
            raise ValueError(f"Sign name {s['name']!r} is not a string: put it in quotes in signs.yaml")
    names = [s["name"] for s in cfg["signs"]]
    for n in names:
        if not SIGN_NAME_RE.match(n):
            raise ValueError(f"Invalid sign name {n!r}: use lowercase_with_underscores")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate sign names in signs.yaml")
    return cfg


def sign_names(cfg: dict) -> list[str]:
    return [str(s["name"]) for s in cfg["signs"]]
