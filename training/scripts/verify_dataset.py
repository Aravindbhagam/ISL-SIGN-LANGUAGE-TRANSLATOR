"""Step 1.2 check: is data/raw/ complete and readable?

Usage (from the repo root):
    python training/scripts/verify_dataset.py

ERRORS (exit code 1): missing sign folders, bad filenames, unreadable or
too-short videos, folders not listed in signs.yaml.
WARNINGS: signs with few clips, or recorded by only one signer.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

import cv2

from common import CLIP_NAME_RE, RAW_DIR, load_config, sign_names

MIN_CLIPS = 15       # below this, a class is unlikely to train reliably
MIN_SECONDS = 0.8    # shorter than any real sign; probably a broken file


def main() -> int:
    names = sign_names(load_config())
    errors: list[str] = []
    warnings: list[str] = []
    rows = []

    if RAW_DIR.is_dir():
        extra = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir() and d.name not in names)
        errors += [f"folder '{e}' is not in signs.yaml" for e in extra]

    for sign in names:
        folder = RAW_DIR / sign
        if not folder.is_dir():
            errors.append(f"{sign}: folder missing")
            rows.append((sign, 0, "", ""))
            continue
        sources: Counter[str] = Counter()
        signers: defaultdict[str, int] = defaultdict(int)
        for f in sorted(folder.iterdir()):
            if f.name.startswith(".") or f.name.endswith(".part.mp4"):
                continue
            m = CLIP_NAME_RE.match(f.name)
            if not m:
                errors.append(f"{sign}/{f.name}: name must be <source>_<signer>_<id>.mp4")
                continue
            cap = cv2.VideoCapture(str(f))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            readable = cap.read()[0]
            cap.release()
            seconds = frames / fps if fps > 0 else 0
            if not readable or seconds < MIN_SECONDS:
                errors.append(f"{sign}/{f.name}: unreadable or too short ({seconds:.2f}s)")
                continue
            sources[m["source"]] += 1
            signers[f"{m['source']}:{m['signer']}"] += 1

        total = sum(sources.values())
        if total < MIN_CLIPS:
            warnings.append(f"{sign}: only {total} clips (aim for >= {MIN_CLIPS})")
        if len(signers) < 2 and total:
            warnings.append(f"{sign}: only one signer; the model may learn that person, not the sign")
        rows.append((
            sign, total,
            ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
            ", ".join(f"{k}={v}" for k, v in sorted(signers.items())),
        ))

    print(f"{'sign':<14}{'clips':>6}  {'by source':<22}by signer")
    for sign, total, src, sgn in rows:
        print(f"{sign:<14}{total:>6}  {src:<22}{sgn}")
    print(f"{'TOTAL':<14}{sum(r[1] for r in rows):>6}\n")
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print("\nDataset OK." if not errors else f"\n{len(errors)} error(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
