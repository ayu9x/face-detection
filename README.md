# Face Detection Suite

A feature-rich face detection application built with OpenCV and Python. Detects faces in images, video files, and live webcam streams with optional age/gender prediction, eye/smile detection, face blurring, video recording, and more.

---

## Features

| Feature | Description |
|---|---|
| **Face Detection** | Haar Cascade + DNN (SSD) dual-mode detection |
| **Age Prediction** | Estimates age range using a Caffe deep learning model |
| **Gender Prediction** | Predicts male/female using a Caffe deep learning model |
| **Eye Detection** | Detects eyes within each face using Haar cascades |
| **Smile Detection** | Detects smiles within each face using Haar cascades |
| **Face Blurring** | Privacy mode — blurs all detected faces in real time |
| **Screenshot Capture** | Save any frame as a JPEG during video mode |
| **Video Recording** | Record the annotated video output to an AVI file |
| **FPS Counter** | Real-time performance overlay |
| **Batch Processing** | Process all images in a folder at once |
| **Face Cropping** | Automatically saves cropped face images |
| **Detection Logging** | Logs detection stats to a CSV file for analytics |
| **JSON Configuration** | Customize defaults via a `config.json` file |
| **Interactive Controls** | Toggle every feature on/off with keyboard shortcuts |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/face-detection.git
cd face-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download models

```bash
python face.py --download-models
```

This downloads the age prediction, gender prediction, and DNN face detector models into the `models/` folder.

---

## Quick Start

### Webcam — all features

```bash
python face.py --video --all
```

### Single image

```bash
python face.py --image photo.jpg --age --gender
```

### Video file

```bash
python face.py --video-file clip.mp4 --age --eyes
```

### Privacy blur mode

```bash
python face.py --video --blur
```

### Batch process a folder

```bash
python face.py --folder ./photos --age --gender --smile
```

---

## Usage

```
python face.py [OPTIONS]
```

### Input Sources (pick one)

| Flag | Description |
|---|---|
| `--image PATH` | Detect faces in a single image |
| `--folder DIR` | Process all images in a directory |
| `--video` | Use webcam for live detection |
| `--video-file PATH` | Process a video file |
| `--camera N` | Camera index (default: 0) |

### Features

| Flag | Description |
|---|---|
| `--age` | Enable age prediction |
| `--gender` | Enable gender prediction |
| `--eyes` | Enable eye detection |
| `--smile` | Enable smile detection |
| `--blur` | Blur detected faces (privacy) |
| `--dnn` | Use DNN face detector instead of Haar |
| `--all` | Enable age + gender + eyes + smile |

### Detection Parameters

| Flag | Default | Description |
|---|---|---|
| `--scale-factor` | 1.1 | Haar cascade scale factor |
| `--min-neighbors` | 5 | Haar cascade min neighbors |
| `--confidence` | 0.6 | DNN confidence threshold |

### Output

| Flag | Description |
|---|---|
| `--output PATH` | Save result to specific path (image mode) |
| `--output-dir DIR` | Output directory (default: `output/`) |
| `--no-display` | Headless mode (no GUI windows) |

### Misc

| Flag | Description |
|---|---|
| `--config FILE` | Load settings from a JSON config file |
| `--download-models` | Download all required DNN models |
| `--verbose` | Enable debug logging |

---

## Keyboard Shortcuts (Video Mode)

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save screenshot |
| `r` | Start / stop recording |
| `b` | Toggle face blur |
| `a` | Toggle age prediction |
| `g` | Toggle gender prediction |
| `e` | Toggle eye detection |
| `m` | Toggle smile detection |
| `f` | Toggle FPS display |
| `d` | Switch between Haar / DNN detection |
| `+` / `-` | Adjust detection sensitivity |
| `p` | Pause / resume |
| `h` | Show / hide help overlay |

---

## Configuration

You can customize defaults via `config.json`:

```json
{
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_face_size": [30, 30],
    "dnn_confidence_threshold": 0.6,
    "use_dnn": false,
    "predict_age": false,
    "predict_gender": false,
    "detect_eyes": false,
    "detect_smile": false,
    "blur_faces": false,
    "show_fps": true,
    "output_dir": "output",
    "blur_strength": 99
}
```

Load it with:

```bash
python face.py --video --config config.json
```

---

## Output Structure

```
output/
├── detection_log.csv        # Detection analytics log
├── photo_detected_*.jpg     # Annotated images
├── face_crops/              # Individual cropped faces
│   ├── face_*_0_(25-32)_Male.jpg
│   └── ...
├── screenshots/             # Manual screenshots
│   └── screenshot_*.jpg
└── recordings/              # Video recordings
    └── recording_*.avi
```

---

## Models

| Model | Purpose | Size |
|---|---|---|
| `age_net.caffemodel` | Age prediction | ~44 MB |
| `age_deploy.prototxt` | Age model config | ~3 KB |
| `gender_net.caffemodel` | Gender prediction | ~44 MB |
| `gender_deploy.prototxt` | Gender model config | ~3 KB |
| `res10_300x300_ssd_iter_140000.caffemodel` | DNN face detection | ~10 MB |
| `deploy.prototxt` | DNN model config | ~28 KB |

All models are downloaded automatically with `--download-models`.

---

## Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy
- gdown (for model downloads)

---

## Examples

```bash
# Basic webcam detection
python face.py --video

# Full analysis on webcam
python face.py --video --all

# Analyze a photo with age and gender
python face.py --image portrait.jpg --age --gender

# Privacy mode on video file
python face.py --video-file meeting.mp4 --blur

# DNN-based detection (more accurate, slower)
python face.py --video --dnn --age

# Batch process with all features
python face.py --folder ./team_photos --all --no-display

# Use custom config
python face.py --video --config config.json
```

---

## License

MIT License — feel free to use, modify, and distribute.
