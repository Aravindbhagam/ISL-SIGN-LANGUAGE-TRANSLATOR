# Training pipeline

Python code that turns sign videos into a TFLite model for the Android app.

## Setup (step 1.1)

### macOS (Apple Silicon: M1–M4)

```bash
brew install python@3.11                                   # needs Homebrew: https://brew.sh
python3.11 -c "import platform; print(platform.machine())" # must print: arm64
```

We deliberately do **not** install `tensorflow-metal` (GPU plugin): the model
is tiny, CPU training takes minutes, and the plugin is often out of sync with
TensorFlow releases. When you record clips (step 1.2), macOS will ask to give
your terminal camera access: allow it in System Settings → Privacy & Security → Camera.

### All platforms

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

## Dataset (step 1.2)

Final layout (git-ignored, never committed):

```
training/data/raw/<sign>/<source>_<signer>_<id>.mp4
    e.g. raw/hello/include_unk_MVI5186-3fa2c1d0.mp4
         raw/hello/self_s01_20260923-101500-03.mp4
```

The vocabulary lives in `config/signs.yaml`, which is the single source of truth.

### 1. INCLUDE (AI4Bharat)

INCLUDE is published on Zenodo (search "INCLUDE Indian Sign Language
Zenodo", or follow the link from https://ai4bharat.iitm.ac.in). It is split
into per-category zips and is tens of GB in total, so **download only the
categories you need** (e.g. Greetings, Occupations). Check the license shown
on the Zenodo page and cite the paper: Sridhar et al., *INCLUDE: A Large
Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020.

```bash
# unzip the categories into one folder, e.g. ~/Datasets/INCLUDE, then:
python training/scripts/organize_include.py --src ~/Datasets/INCLUDE --list
# put the exact labels into `include:` in config/signs.yaml, then:
python training/scripts/organize_include.py --src ~/Datasets/INCLUDE
```

### 2. Your own clips

```bash
python training/scripts/record_clips.py --signer s01
```

Tips that measurably matter for the model:
- Frame yourself from the waist up, with both hands in view the whole time.
- Vary the conditions across sessions (lighting, background, clothes,
  distance). The phone will see all of these.
- Start and end each clip with your hands resting, like a real user would.
- Record `idle` too: rest, scratch your face, fix your hair, but don't sign.
- Each person gets their own `--signer` ID. More signers help more than more clips.

### 3. Verify

```bash
python training/scripts/verify_dataset.py
```
