"""Step 1.1 check: verify the training environment is usable.

Run from the repo root:  python training/scripts/check_env.py
Exits non-zero if anything is wrong, so CI can use it later too.
"""

import sys

REQUIRED_PYTHON = (3, 11)


def main() -> int:
    ok = True

    if sys.version_info[:2] != REQUIRED_PYTHON:
        print(f"[FAIL] Python {sys.version.split()[0]}; expected 3.11.x")
        ok = False
    else:
        print(f"[ OK ] Python {sys.version.split()[0]}")

    import cv2
    import matplotlib
    import mediapipe as mp
    import numpy as np
    import sklearn
    import tensorflow as tf
    import yaml

    for name, mod in [
        ("numpy", np), ("opencv", cv2), ("mediapipe", mp),
        ("tensorflow", tf), ("scikit-learn", sklearn),
        ("matplotlib", matplotlib), ("pyyaml", yaml),
    ]:
        print(f"[ OK ] {name:<13} {mod.__version__}")

    # Step 1.3 depends on the legacy Holistic solution; fail early if missing.
    try:
        holistic = mp.solutions.holistic.Holistic(static_image_mode=True)
        holistic.process(np.zeros((64, 64, 3), dtype=np.uint8))
        holistic.close()
        print("[ OK ] MediaPipe Holistic runs")
    except Exception as e:  # noqa: BLE001 - report any failure clearly
        print(f"[FAIL] MediaPipe Holistic: {e}")
        ok = False

    # Step 1.7 depends on TFLite conversion of an LSTM; smoke-test it now.
    # batch_size=1 is deliberate: with a dynamic batch dim, Keras 3 LSTMs lower
    # to TensorList ops that TFLite builtins can't handle. The phone only ever
    # classifies one window at a time, so a fixed batch of 1 costs nothing.
    try:
        model = tf.keras.Sequential([
            tf.keras.Input((30, 8), batch_size=1),
            tf.keras.layers.LSTM(4),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        tflite = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        print(f"[ OK ] TFLite conversion of an LSTM ({len(tflite)} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] TFLite conversion: {e}")
        ok = False

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPUs visible to TensorFlow: {len(gpus)} (CPU is fine for this model)")

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
