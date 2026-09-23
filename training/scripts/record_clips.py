"""Record your own sign clips with the webcam into data/raw/<sign>/.

Usage (from the repo root):
    python training/scripts/record_clips.py --signer s01
    python training/scripts/record_clips.py --signer s01 --signs hello,water

Controls in the preview window:
    SPACE  record the next clip (after a countdown)
    R      delete the last clip you recorded (bad take) and redo it
    N      skip to the next sign
    Q/ESC  quit (everything recorded so far is kept)

Each person who records gets their own --signer ID (s01, s02, ...). Never
reuse an ID for a different person: step 1.5 splits data by signer, and a
wrong ID silently leaks the same person into both train and test.

The preview is mirrored (feels natural, like a mirror) but the SAVED video
is not, which matches the raw frames the phone's front camera gives the
model. Saving mirrored video would swap left and right hands.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from common import RAW_DIR, load_config, sign_names

WINDOW = "record_clips (SPACE record, R redo, N next sign, Q quit)"
RED, GREEN, WHITE = (0, 0, 255), (0, 200, 0), (255, 255, 255)


def put(frame, text, y, color=WHITE, scale=0.9):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def show(frame, lines, color=WHITE):
    view = cv2.flip(frame, 1)  # mirror for display only
    for i, line in enumerate(lines):
        put(view, line, 40 + 40 * i, color)
    cv2.imshow(WINDOW, view)
    return cv2.waitKey(1) & 0xFF


def existing(sign: str, signer: str) -> list[Path]:
    return sorted((RAW_DIR / sign).glob(f"self_{signer}_*.mp4"))


def record_clip(cap, sign: str, seconds: float, countdown: float, header: str) -> list:
    t0 = time.monotonic()
    while (left := countdown - (time.monotonic() - t0)) > 0:
        ok, frame = cap.read()
        if ok and show(frame, [header, f"Get ready: {left:.1f}s"], GREEN) in (ord("q"), 27):
            return []
    frames, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((time.monotonic() - t0, frame))
        show(frame, [header, f"RECORDING  {sign.upper()}"], RED)
    return frames


def save(frames: list, path: Path) -> None:
    # Webcams (especially on macOS) often report a wrong FPS, so we write the
    # FPS we actually measured. Otherwise the clip plays too fast/slow and the
    # 30-frame sampling in step 1.3 covers the wrong amount of time.
    duration = frames[-1][0] - frames[0][0]
    fps = (len(frames) - 1) / duration if duration > 0 else 30.0
    h, w = frames[0][1].shape[:2]
    tmp = path.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _, f in frames:
        writer.write(f)
    writer.release()
    tmp.rename(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signer", required=True, help="Signer ID, e.g. s01 (lowercase letters/digits)")
    ap.add_argument("--signs", help="Comma-separated subset of signs (default: all in signs.yaml)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if not args.signer.isalnum() or args.signer != args.signer.lower():
        print("--signer must be lowercase letters/digits, e.g. s01")
        return 1
    cfg = load_config()
    rec = cfg["record"]
    signs = sign_names(cfg)
    if args.signs:
        wanted = [s.strip() for s in args.signs.split(",")]
        unknown = set(wanted) - set(signs)
        if unknown:
            print(f"Not in signs.yaml: {sorted(unknown)}")
            return 1
        signs = wanted

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened() or not cap.read()[0]:
        print("Cannot open the webcam. On macOS: System Settings > Privacy & Security >"
              " Camera > allow your terminal app, then restart the terminal.")
        return 1

    last_saved: Path | None = None
    try:
        for sign in signs:
            (RAW_DIR / sign).mkdir(parents=True, exist_ok=True)
            while (n := len(existing(sign, args.signer))) < rec["clips_per_sign"]:
                header = f"{sign.upper()}  clip {n + 1}/{rec['clips_per_sign']}  (signer {args.signer})"
                ok, frame = cap.read()
                if not ok:
                    continue
                key = show(frame, [header, "SPACE = record"])
                if key in (ord("q"), 27):
                    return 0
                if key == ord("n"):
                    break
                if key == ord("r") and last_saved and last_saved.exists():
                    last_saved.unlink()
                    print(f"deleted {last_saved.name}")
                    last_saved = None
                if key == ord(" "):
                    frames = record_clip(cap, sign, rec["clip_seconds"], rec["countdown_seconds"], header)
                    if len(frames) < 10:
                        continue
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    last_saved = RAW_DIR / sign / f"self_{args.signer}_{stamp}-{n:02d}.mp4"
                    save(frames, last_saved)
                    print(f"saved {last_saved.relative_to(RAW_DIR)} ({len(frames)} frames)")
        print("All requested signs have enough clips.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
