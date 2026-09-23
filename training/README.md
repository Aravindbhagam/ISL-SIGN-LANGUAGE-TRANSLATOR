# Training pipeline

Python code that turns sign videos into a TFLite model for the Android app.

## Setup (step 1.1)

Requires **Python 3.11, 64-bit**. (MediaPipe and TensorFlow publish wheels for
specific Python versions; 3.11 is supported by both pinned versions. 3.13 is not.)

```bash
# from the repo root
python3.11 -m venv .venv            # Windows: py -3.11 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r training/requirements.txt
python training/scripts/check_env.py
```

Expected: every line `[ OK ]` and `All checks passed.`
