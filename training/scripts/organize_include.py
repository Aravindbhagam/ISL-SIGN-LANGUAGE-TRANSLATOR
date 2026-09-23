"""Import the chosen INCLUDE videos into data/raw/<sign>/.

INCLUDE (AI4Bharat) is laid out roughly as
    <Category>/<NN>. <Word>/<video>.MOV      e.g. Greetings/48. Hello/MVI_5186.MOV
This script finds word folders at any depth, so it does not depend on the
exact category nesting of your download.

Usage (from the repo root):
    # 1. See which word labels your download actually contains
    python training/scripts/organize_include.py --src ~/Downloads/INCLUDE --list
    # 2. Put the labels you want into config/signs.yaml (`include:`), then
    python training/scripts/organize_include.py --src ~/Downloads/INCLUDE

Videos are re-encoded to .mp4 and downscaled to at most --max-height pixels.
Tradeoff: re-encoding loses a little quality, but INCLUDE is 1080p and the
phone camera will give us ~480-720p frames, so training on similar
resolutions reduces train/phone mismatch and cuts disk use several-fold.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from common import RAW_DIR, load_config

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
WORD_DIR_RE = re.compile(r"^\d+\.\s*(?P<label>.+)$")


def normalize(label: str) -> str:
    """Case/space-insensitive key so 'Thank you' matches 'thank  You'."""
    return " ".join(label.lower().split())


def find_word_dirs(src: Path) -> dict[str, list[Path]]:
    """Map normalized INCLUDE label -> list of video files (all categories)."""
    found: dict[str, list[Path]] = defaultdict(list)
    for d in sorted(p for p in src.rglob("*") if p.is_dir()):
        m = WORD_DIR_RE.match(d.name)
        if not m:
            continue
        videos = sorted(f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        found[normalize(m.group("label"))].extend(videos)
    return found


def clip_id(video: Path, src: Path) -> str:
    """Stable, filename-safe ID. The hash keeps IDs unique when two
    categories contain files with the same name (e.g. MVI_0001.MOV)."""
    rel = video.relative_to(src).as_posix()
    stem = re.sub(r"[^A-Za-z0-9]", "", video.stem)[:20]
    return f"{stem}-{hashlib.sha1(rel.encode()).hexdigest()[:8]}"


def transcode(src: Path, dst: Path, max_height: int) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    frames = 0
    tmp = dst.with_suffix(".part.mp4")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if h > max_height:
            scale = max_height / h
            # Even dimensions: many H.264/MPEG-4 decoders reject odd sizes.
            frame = cv2.resize(frame, (int(w * scale) // 2 * 2, max_height), interpolation=cv2.INTER_AREA)
        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        writer.write(frame)
        frames += 1
    cap.release()
    if writer is not None:
        writer.release()
    if frames == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dst)  # write-then-rename: an interrupted run never leaves a half file
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Folder containing the extracted INCLUDE zips")
    ap.add_argument("--list", action="store_true", help="List available labels and exit")
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        print(f"Not a folder: {src}")
        return 1
    found = find_word_dirs(src)
    if not found:
        print(f"No '<NN>. <Word>' folders found under {src}. Did you extract the zips?")
        return 1

    if args.list:
        for label in sorted(found):
            print(f"{len(found[label]):4d}  {label}")
        print(f"\n{len(found)} labels, {sum(map(len, found.values()))} videos")
        return 0

    cfg = load_config()
    missing = []
    plan: list[tuple[Path, Path]] = []
    for sign in cfg["signs"]:
        for label in sign.get("include") or []:
            videos = found.get(normalize(str(label)))
            if not videos:
                missing.append(f"{sign['name']}: '{label}'")
                continue
            out_dir = RAW_DIR / str(sign["name"])
            for v in videos:
                # INCLUDE does not expose signer IDs in its filenames, so they
                # are tagged 'unk'. See step 1.5 for how this affects splitting.
                plan.append((v, out_dir / f"include_unk_{clip_id(v, src)}.mp4"))

    if missing:
        print("These labels are in signs.yaml but not in your download:")
        print("\n".join(f"  - {m}" for m in missing))
        print("Run with --list and fix the spelling, or download that category.")
        return 1
    if not plan:
        print("No `include:` labels set in config/signs.yaml. Nothing to do.")
        return 0

    done = skipped = failed = 0
    for i, (v, dst) in enumerate(plan, 1):
        if dst.exists():
            skipped += 1
            continue
        print(f"[{i}/{len(plan)}] {v.relative_to(src)} -> {dst.relative_to(RAW_DIR)}")
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if transcode(v, dst, args.max_height):
            done += 1
        else:
            failed += 1
            print(f"  FAILED to read {v}")
    print(f"\nconverted {done}, already present {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
