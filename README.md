# Driver Drowsiness Detection

A real-time driver drowsiness detection system built with Python, OpenCV and dlib. It watches the driver's eyes through a webcam and sounds an alarm when the eyes stay closed for too long.

## How it works

1. **Face detection**: dlib's frontal face detector finds the face, with an OpenCV Haar cascade as a fallback.
2. **Eye tracking**: dlib's 68-point facial landmark model locates both eyes. If the landmark model is missing, the program falls back to Haar cascade eye detection.
3. **Drowsiness check**: the Eye Aspect Ratio (EAR) is computed for each eye and smoothed over recent frames to avoid flicker.
4. **Alert**: if both eyes stay closed for 15 consecutive frames, a red warning appears on screen and an alarm sound plays.

## Project structure

```
my_drowsiness_detection.py                 Main script
alarm.wav                                  Alarm sound
haarcascade_frontalface_alt.xml            Face detector
haarcascade_lefteye_2splits.xml            Left eye detector (fallback)
haarcascade_righteye_2splits.xml           Right eye detector (fallback)
shape_predictor_68_face_landmarks.dat      dlib landmark model (download, see below)
simple.py                                  Experimental CNN model definition (not used by the main script)
requirements.txt                           Python dependencies
setup_project.bat                          One-click Windows setup
run_detection.bat                          One-click Windows launcher
```

## Getting started

### Requirements

- Python 3.10
- A webcam
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda (recommended, since dlib installs most easily through conda)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

**Windows (easiest):** double-click `setup_project.bat`.

**Manual setup:**

```bash
conda create -n drowsiness_detection python=3.10 -y
conda activate drowsiness_detection
conda install -c conda-forge dlib -y
pip install -r requirements.txt
```

### Download the landmark model

The dlib landmark model is about 95 MB, so it is not stored in this repository.

1. Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib.net/files](http://dlib.net/files/).
2. Extract it and place `shape_predictor_68_face_landmarks.dat` in the project folder.

Without it, the program still runs but uses the less accurate Haar cascade fallback.

### Run

**Windows:** double-click `run_detection.bat`.

**Any platform:**

```bash
conda activate drowsiness_detection
python my_drowsiness_detection.py
```

Press **q** in the video window to quit.

## Configuration

Edit these constants near the top of `my_drowsiness_detection.py`:

| Constant | Default | Meaning |
|---|---|---|
| `EYE_AR_THRESH` | `0.2` | Eye aspect ratio below which an eye counts as closed |
| `EYE_AR_CONSEC_FRAMES` | `15` | Consecutive closed-eye frames before the alarm triggers |

Lower the frame count for a more sensitive alarm, raise it to reduce false alarms from normal blinking.

## Troubleshooting

- **Camera won't open:** close other apps that use the webcam, and check camera permissions in your OS settings.
- **No sound:** make sure `alarm.wav` is in the project folder and your audio output is working.
- **Poor detection:** use good, even lighting and keep your face roughly centered and facing the camera.

## Technologies

OpenCV, dlib, NumPy, pygame, Python.

Project Type
BCA Mini Project
