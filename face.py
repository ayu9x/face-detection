"""
Face Detection Suite - Advanced face detection with age/gender prediction,
eye/smile detection, face blurring, and more.

Features:
  - Face detection (Haar Cascade + DNN)
  - Age prediction
  - Gender prediction
  - Eye & Smile detection
  - Face blurring (privacy mode)
  - Video file & webcam support
  - Screenshot capture (press 's')
  - Video recording (press 'r')
  - FPS counter & stats overlay
  - Face ROI extraction & saving
  - Configurable via CLI and JSON config

Controls (video mode):
  q - Quit
  s - Save screenshot
  r - Start/stop recording
  b - Toggle face blur mode
  a - Toggle age prediction
  g - Toggle gender prediction
  e - Toggle eye detection
  m - Toggle smile detection
  f - Toggle FPS display
  d - Toggle detection method (Haar/DNN)
  + - Increase detection sensitivity
  - - Decrease detection sensitivity
  p - Pause/resume
  h - Toggle help overlay

Author: Face Detection Suite
"""

import cv2
import numpy as np
import argparse
import os
import sys
import json
import time
import logging
from datetime import datetime
from collections import deque

# ─── PyInstaller bundle support ──────────────────────────────────────────────
# When running as a PyInstaller .exe, bundled data is extracted to a temp dir.
# _BASE_DIR points to that temp dir (or the script dir when running normally).

if getattr(sys, 'frozen', False):
    _BASE_DIR = sys._MEIPASS
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FaceDetection")

# ─── Constants ───────────────────────────────────────────────────────────────

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
GENDER_LIST = ["Male", "Female"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Color palette (BGR)
COLORS = {
    "face":       (255, 120, 0),    # blue-ish
    "eye":        (0, 255, 255),    # yellow
    "smile":      (0, 200, 200),    # gold
    "age":        (0, 255, 0),      # green
    "gender":     (255, 0, 255),    # magenta
    "fps":        (0, 255, 0),      # green
    "info":       (255, 255, 255),  # white
    "blur":       (0, 0, 255),      # red
    "panel_bg":   (40, 40, 40),     # dark grey
    "help_bg":    (30, 30, 30),     # darker grey
}

DEFAULT_CONFIG = {
    "scale_factor": 1.1,
    "min_neighbors": 5,
    "min_face_size": [30, 30],
    "dnn_confidence_threshold": 0.6,
    "use_dnn": False,
    "predict_age": False,
    "predict_gender": False,
    "detect_eyes": False,
    "detect_smile": False,
    "blur_faces": False,
    "show_fps": True,
    "camera_id": 0,
    "output_dir": "output",
    "log_file": "detection_log.csv",
    "blur_strength": 99,
    "bbox_thickness": 2,
    "font_scale": 0.55,
}


# ─── Utility helpers ────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file, merged with defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
        logger.info("Loaded config from %s", config_path)
    return cfg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def draw_label(img, text, pos, color, font_scale=0.55, thickness=1, bg=True):
    """Draw text with optional background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    if bg:
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y + 2), COLORS["panel_bg"], -1)
    cv2.putText(img, text, (x + 2, y - 4), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_fancy_rect(img, pt1, pt2, color, thickness=2, corner_len=15):
    """Draw a rectangle with corner accents for a modern look."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Main rectangle
    cv2.rectangle(img, pt1, pt2, color, thickness)
    cl = min(corner_len, (x2 - x1) // 3, (y2 - y1) // 3)
    t = thickness + 1
    # Top-left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, t)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, t)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, t)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, t)


# ─── FPS tracker ─────────────────────────────────────────────────────────────

class FPSTracker:
    """Smooth FPS calculation over a sliding window."""
    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now - self._prev)
        self._prev = now
        return self.fps()

    def fps(self) -> float:
        if not self._times:
            return 0.0
        return len(self._times) / sum(self._times)


# ─── Detection Logger ────────────────────────────────────────────────────────

class DetectionLogger:
    """Log detection events to a CSV file for analytics."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path) if os.path.dirname(log_path) else ".")
        if not os.path.isfile(log_path):
            with open(log_path, "w") as f:
                f.write("timestamp,face_count,ages,genders,source\n")

    def log(self, face_count: int, ages: list, genders: list, source: str = "webcam"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ages_str = "|".join(ages) if ages else ""
        genders_str = "|".join(genders) if genders else ""
        with open(self.log_path, "a") as f:
            f.write(f"{ts},{face_count},{ages_str},{genders_str},{source}\n")


# ─── Main Face Detector Class ────────────────────────────────────────────────

class FaceDetector:
    """
    Full-featured face detector with age/gender prediction,
    eye/smile detection, DNN-based detection, and face blurring.
    """

    def __init__(self, config: dict = None):
        """Initialize the face detector with all models and cascades."""
        self.cfg = config or DEFAULT_CONFIG.copy()

        # ── Haar cascades ─────────────────────────────────────────────
        if getattr(sys, 'frozen', False):
            cascade_dir = os.path.join(_BASE_DIR, 'cv2', 'data') + os.sep
        else:
            cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cascade_dir + "haarcascade_smile.xml"
        )

        # ── DNN face detector (Caffe, ships with OpenCV >= 3.3) ──────
        self.dnn_net = None
        self._init_dnn_face_detector()

        # ── Age & Gender models ──────────────────────────────────────
        self.age_net = None
        self.gender_net = None
        self._init_age_model()
        self._init_gender_model()

        # ── Runtime state ────────────────────────────────────────────
        self.cap = None
        self.fps = FPSTracker()
        self.det_logger = DetectionLogger(
            os.path.join(self.cfg["output_dir"], self.cfg["log_file"])
        )

    # ── Model loaders ────────────────────────────────────────────────

    def _init_dnn_face_detector(self):
        """Load OpenCV DNN face detector (res10 SSD)."""
        proto = os.path.join(_BASE_DIR, "models", "deploy.prototxt")
        model = os.path.join(_BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.isfile(proto) and os.path.isfile(model):
            self.dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
            logger.info("DNN face detector loaded")
        else:
            logger.info("DNN face model not found - using Haar cascade only")

    def _init_age_model(self):
        """Load Caffe age prediction model."""
        proto = os.path.join(_BASE_DIR, "models", "age_deploy.prototxt")
        model = os.path.join(_BASE_DIR, "models", "age_net.caffemodel")
        if os.path.isfile(proto) and os.path.isfile(model):
            self.age_net = cv2.dnn.readNet(model, proto)
            logger.info("Age model loaded")
        else:
            logger.info("Age model not found - age prediction disabled")

    def _init_gender_model(self):
        """Load Caffe gender prediction model."""
        proto = os.path.join(_BASE_DIR, "models", "gender_deploy.prototxt")
        model = os.path.join(_BASE_DIR, "models", "gender_net.caffemodel")
        if os.path.isfile(proto) and os.path.isfile(model):
            self.gender_net = cv2.dnn.readNet(model, proto)
            logger.info("Gender model loaded")
        else:
            logger.info("Gender model not found - gender prediction disabled")

    # ── Prediction helpers ───────────────────────────────────────────

    def predict_age(self, face_img):
        """Predict age range from a cropped face image."""
        if self.age_net is None:
            return None
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.age_net.setInput(blob)
        preds = self.age_net.forward()
        return AGE_BUCKETS[int(np.argmax(preds))]

    def predict_gender(self, face_img):
        """Predict gender from a cropped face image."""
        if self.gender_net is None:
            return None
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()
        return GENDER_LIST[int(np.argmax(preds))]

    # ── Face detection ───────────────────────────────────────────────

    def detect_faces_haar(self, gray):
        """Detect faces using Haar cascade."""
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg["scale_factor"],
            minNeighbors=self.cfg["min_neighbors"],
            minSize=tuple(self.cfg["min_face_size"]),
        )

    def detect_faces_dnn(self, frame):
        """Detect faces using DNN (SSD). Returns list of (x, y, w, h)."""
        if self.dnn_net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.cfg["dnn_confidence_threshold"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def detect_faces(self, frame, gray):
        """Detect faces using the currently selected method."""
        if self.cfg["use_dnn"] and self.dnn_net is not None:
            return self.detect_faces_dnn(frame)
        return self.detect_faces_haar(gray)

    # ── Process a single frame ───────────────────────────────────────

    def process_frame(self, frame, options: dict = None):
        """
        Process a frame: detect faces, annotate, blur, predict age/gender.
        Returns: (annotated_frame, list_of_face_info_dicts)
        """
        opts = {**self.cfg, **(options or {})}
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(frame, gray)

        face_infos = []

        for (x, y, w, h) in faces:
            info = {"bbox": (x, y, w, h)}

            # ── Face blur ────────────────────────────────────────
            if opts.get("blur_faces"):
                roi = output[y:y+h, x:x+w]
                blur_k = opts.get("blur_strength", 99)
                blur_k = blur_k if blur_k % 2 == 1 else blur_k + 1
                output[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (blur_k, blur_k), 30)
                draw_fancy_rect(output, (x, y), (x+w, y+h), COLORS["blur"], opts["bbox_thickness"])
            else:
                draw_fancy_rect(output, (x, y), (x+w, y+h), COLORS["face"], opts["bbox_thickness"])

            face_roi = frame[y:y+h, x:x+w].copy()
            labels = []

            # ── Age prediction ───────────────────────────────────
            if opts.get("predict_age") and self.age_net is not None:
                age = self.predict_age(face_roi)
                if age:
                    info["age"] = age
                    labels.append(f"Age:{age}")

            # ── Gender prediction ────────────────────────────────
            if opts.get("predict_gender") and self.gender_net is not None:
                gender = self.predict_gender(face_roi)
                if gender:
                    info["gender"] = gender
                    labels.append(gender)

            # ── Combined label ───────────────────────────────────
            if labels and not opts.get("blur_faces"):
                label_text = "  ".join(labels)
                draw_label(output, label_text, (x, y - 8), COLORS["age"], opts["font_scale"])

            # ── Eye detection ────────────────────────────────────
            if opts.get("detect_eyes"):
                face_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(
                    face_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
                )
                info["eyes"] = len(eyes)
                for (ex, ey, ew, eh) in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    radius = int(round((ew + eh) * 0.25))
                    cv2.circle(output, center, radius, COLORS["eye"], 2)

            # ── Smile detection ──────────────────────────────────
            if opts.get("detect_smile"):
                face_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(
                    face_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25)
                )
                if len(smiles) > 0:
                    info["smiling"] = True
                    if not opts.get("blur_faces"):
                        draw_label(output, "Smiling :)", (x, y + h + 18), COLORS["smile"], opts["font_scale"])
                else:
                    info["smiling"] = False

            face_infos.append(info)

        return output, face_infos

    # ── Image mode ───────────────────────────────────────────────────

    def detect_from_image(self, image_path: str, save: bool = True):
        """Detect faces in a single image file."""
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Cannot read image: %s", image_path)
            return None, []

        result, face_infos = self.process_frame(img)
        face_count = len(face_infos)
        ages = [f.get("age", "") for f in face_infos if f.get("age")]
        genders = [f.get("gender", "") for f in face_infos if f.get("gender")]
        self.det_logger.log(face_count, ages, genders, source=image_path)

        logger.info("Detected %d face(s) in %s", face_count, image_path)

        # Add summary bar at top
        summary = f"Faces: {face_count}"
        if ages:
            summary += f"  |  Ages: {', '.join(ages)}"
        if genders:
            summary += f"  |  Gender: {', '.join(genders)}"
        draw_label(result, summary, (10, 25), COLORS["info"], 0.6)

        if save:
            ensure_dir(self.cfg["output_dir"])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(self.cfg["output_dir"], f"{base}_detected_{ts}.jpg")
            cv2.imwrite(out_path, result)
            logger.info("Saved result to %s", out_path)

            # Also save individual face crops
            self._save_face_crops(img, face_infos)

        return result, face_infos

    def _save_face_crops(self, frame, face_infos):
        """Save cropped face regions as individual image files."""
        crops_dir = os.path.join(self.cfg["output_dir"], "face_crops")
        ensure_dir(crops_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, info in enumerate(face_infos):
            x, y, w, h = info["bbox"]
            crop = frame[y:y+h, x:x+w]
            parts = [f"face_{ts}_{i}"]
            if "age" in info:
                parts.append(info["age"])
            if "gender" in info:
                parts.append(info["gender"])
            name = "_".join(parts) + ".jpg"
            cv2.imwrite(os.path.join(crops_dir, name), crop)

    # ── Batch image mode ─────────────────────────────────────────────

    def detect_from_folder(self, folder_path: str):
        """Run detection on every image in a folder."""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        files = sorted(
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in exts
        )
        logger.info("Processing %d images from %s", len(files), folder_path)
        total_faces = 0
        for fname in files:
            path = os.path.join(folder_path, fname)
            _, infos = self.detect_from_image(path)
            total_faces += len(infos)
        logger.info("Batch complete - %d total faces in %d images", total_faces, len(files))

    # ── Video / webcam mode ──────────────────────────────────────────

    def start_video(self, source=0):
        """
        Start face detection on a video source (webcam index or file path).
        Interactive controls via keyboard.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            logger.error("Cannot open video source: %s", source)
            return

        source_name = f"camera_{source}" if isinstance(source, int) else os.path.basename(str(source))
        logger.info("Video started - source: %s", source_name)

        # Video writer for recording
        writer = None
        recording = False
        paused = False
        show_help = False
        frame_count = 0
        log_interval = 30  # log every N frames

        ensure_dir(self.cfg["output_dir"])

        # Dynamic toggles (mutable copy of cfg)
        opts = {**self.cfg}

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
            else:
                # Show pause overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Press 'p' to resume", (frame.shape[1] // 2 - 120, frame.shape[0] // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["info"], 1, cv2.LINE_AA)
                cv2.imshow("Face Detection Suite", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            current_fps = self.fps.tick()
            frame_count += 1

            # Process frame
            output, face_infos = self.process_frame(frame, opts)

            # ── Info panel (top bar) ─────────────────────────────
            panel_h = 36
            cv2.rectangle(output, (0, 0), (output.shape[1], panel_h), COLORS["panel_bg"], -1)

            face_count = len(face_infos)
            status_parts = [f"Faces: {face_count}"]
            if opts.get("show_fps"):
                status_parts.append(f"FPS: {current_fps:.1f}")
            if opts.get("use_dnn") and self.dnn_net:
                status_parts.append("DNN")
            else:
                status_parts.append("Haar")
            if opts.get("blur_faces"):
                status_parts.append("BLUR")
            if opts.get("predict_age"):
                status_parts.append("AGE")
            if opts.get("predict_gender"):
                status_parts.append("GENDER")
            if opts.get("detect_eyes"):
                status_parts.append("EYES")
            if opts.get("detect_smile"):
                status_parts.append("SMILE")
            if recording:
                status_parts.append("REC")
                # Blinking red dot
                if frame_count % 20 < 10:
                    cv2.circle(output, (output.shape[1] - 20, 18), 8, (0, 0, 255), -1)

            status_text = "  |  ".join(status_parts)
            cv2.putText(output, status_text, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["info"], 1, cv2.LINE_AA)

            # ── Hint for help ────────────────────────────────────
            if not show_help and frame_count < 150:
                cv2.putText(output, "Press 'h' for shortcuts",
                            (output.shape[1] - 220, output.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

            # ── Help overlay ─────────────────────────────────────
            if show_help:
                self._draw_help_overlay(output)

            # ── Recording ────────────────────────────────────────
            if recording and writer is not None:
                writer.write(output)

            # ── Periodic logging ─────────────────────────────────
            if frame_count % log_interval == 0 and face_count > 0:
                ages = [f.get("age", "") for f in face_infos if f.get("age")]
                genders = [f.get("gender", "") for f in face_infos if f.get("gender")]
                self.det_logger.log(face_count, ages, genders, source=source_name)

            # ── Display ──────────────────────────────────────────
            cv2.imshow("Face Detection Suite", output)

            # ── Keyboard handling ────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                self._save_screenshot(output)
            elif key == ord("r"):
                if not recording:
                    recording, writer = self._start_recording(frame)
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    logger.info("Recording stopped")
            elif key == ord("b"):
                opts["blur_faces"] = not opts.get("blur_faces", False)
                logger.info("Blur mode: %s", "ON" if opts["blur_faces"] else "OFF")
            elif key == ord("a"):
                opts["predict_age"] = not opts.get("predict_age", False)
                logger.info("Age prediction: %s", "ON" if opts["predict_age"] else "OFF")
            elif key == ord("g"):
                opts["predict_gender"] = not opts.get("predict_gender", False)
                logger.info("Gender prediction: %s", "ON" if opts["predict_gender"] else "OFF")
            elif key == ord("e"):
                opts["detect_eyes"] = not opts.get("detect_eyes", False)
                logger.info("Eye detection: %s", "ON" if opts["detect_eyes"] else "OFF")
            elif key == ord("m"):
                opts["detect_smile"] = not opts.get("detect_smile", False)
                logger.info("Smile detection: %s", "ON" if opts["detect_smile"] else "OFF")
            elif key == ord("f"):
                opts["show_fps"] = not opts.get("show_fps", True)
            elif key == ord("d"):
                if self.dnn_net is not None:
                    opts["use_dnn"] = not opts.get("use_dnn", False)
                    logger.info("Detection: %s", "DNN" if opts["use_dnn"] else "Haar")
                else:
                    logger.warning("DNN model not available. Run: python face.py --download-models")
            elif key == ord("+") or key == ord("="):
                opts["min_neighbors"] = max(1, opts["min_neighbors"] - 1)
                logger.info("Sensitivity up (min_neighbors=%d)", opts["min_neighbors"])
            elif key == ord("-"):
                opts["min_neighbors"] = opts["min_neighbors"] + 1
                logger.info("Sensitivity down (min_neighbors=%d)", opts["min_neighbors"])
            elif key == ord("p"):
                paused = True
            elif key == ord("h"):
                show_help = not show_help

        # Cleanup
        if writer is not None:
            writer.release()
        self.stop_video()

    def _draw_help_overlay(self, frame):
        """Draw translucent help overlay with keyboard shortcuts."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        bx1, by1 = w // 4, h // 6
        bx2, by2 = 3 * w // 4, 5 * h // 6
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), COLORS["help_bg"], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLORS["face"], 1)

        lines = [
            "KEYBOARD SHORTCUTS",
            "",
            "q   - Quit",
            "s   - Save screenshot",
            "r   - Start/stop recording",
            "b   - Toggle face blur",
            "a   - Toggle age prediction",
            "g   - Toggle gender prediction",
            "e   - Toggle eye detection",
            "m   - Toggle smile detection",
            "f   - Toggle FPS display",
            "d   - Toggle Haar / DNN detection",
            "+/- - Adjust sensitivity",
            "p   - Pause / resume",
            "h   - Close this help",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_off = by1 + 30
        for line in lines:
            scale = 0.7 if line == lines[0] else 0.5
            color = COLORS["face"] if line == lines[0] else COLORS["info"]
            thickness = 2 if line == lines[0] else 1
            cv2.putText(frame, line, (bx1 + 20, y_off), font, scale, color, thickness, cv2.LINE_AA)
            y_off += 26

    def _save_screenshot(self, frame):
        """Save the current frame as a screenshot."""
        screenshot_dir = os.path.join(self.cfg["output_dir"], "screenshots")
        ensure_dir(screenshot_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(screenshot_dir, f"screenshot_{ts}.jpg")
        cv2.imwrite(path, frame)
        logger.info("Screenshot saved: %s", path)

    def _start_recording(self, frame):
        """Start recording video output."""
        recordings_dir = os.path.join(self.cfg["output_dir"], "recordings")
        ensure_dir(recordings_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(recordings_dir, f"recording_{ts}.avi")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
        logger.info("Recording started: %s", path)
        return True, writer

    def stop_video(self):
        """Release video capture and close windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


# ─── Model downloader ────────────────────────────────────────────────────────

def download_models():
    """Download all required model files if they don't exist."""
    import urllib.request

    ensure_dir("models")

    # Small prototxt files (direct download)
    prototxt_models = {
        "models/age_deploy.prototxt": (
            "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning"
            "/master/age_net_definitions/deploy.prototxt"
        ),
        "models/gender_deploy.prototxt": (
            "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning"
            "/master/gender_net_definitions/deploy.prototxt"
        ),
        "models/deploy.prototxt": (
            "https://raw.githubusercontent.com/opencv/opencv"
            "/master/samples/dnn/face_detector/deploy.prototxt"
        ),
    }

    for local_path, url in prototxt_models.items():
        if not os.path.isfile(local_path):
            logger.info("Downloading %s ...", local_path)
            try:
                urllib.request.urlretrieve(url, local_path)
                logger.info("  Done: %s", local_path)
            except Exception as e:
                logger.warning("  Failed to download %s: %s", local_path, e)

    # Larger caffemodel files (via gdown from Google Drive)
    large_models = {
        "models/age_net.caffemodel": "1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW",
        "models/gender_net.caffemodel": "1W_moLzMlGiELyPxWiYQJ9KFaXroQ1T7Z",
        "models/res10_300x300_ssd_iter_140000.caffemodel": "1weGAqSPMgKFSVlMdBwHUqB7r5h8Cy5OA",
    }

    missing_large = {k: v for k, v in large_models.items() if not os.path.isfile(k)}
    if missing_large:
        try:
            import gdown
            for local_path, file_id in missing_large.items():
                logger.info("Downloading %s via gdown ...", local_path)
                try:
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        local_path, quiet=False
                    )
                    logger.info("  Done: %s", local_path)
                except Exception as e:
                    logger.warning("  Failed: %s", e)
        except ImportError:
            logger.warning(
                "gdown not installed. Install with: pip install gdown\n"
                "Missing models: %s", ", ".join(missing_large.keys())
            )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face.py",
        description="Face Detection Suite - detect, analyze & annotate faces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python face.py --video                        Webcam with default settings\n"
            "  python face.py --video --age --gender --eyes  Full feature webcam mode\n"
            "  python face.py --image photo.jpg --age        Detect faces in a photo\n"
            "  python face.py --folder ./photos --age        Batch process a folder\n"
            "  python face.py --video --blur                 Privacy blur mode\n"
            "  python face.py --video-file clip.mp4 --age    Analyze a video file\n"
            "  python face.py --video --all                  All features enabled\n"
            "  python face.py --download-models              Download all DNN models\n"
        ),
    )

    # Input sources
    src = p.add_argument_group("Input source (pick one)")
    src.add_argument("--image", metavar="PATH", help="Path to input image")
    src.add_argument("--folder", metavar="DIR", help="Process all images in a folder")
    src.add_argument("--video", action="store_true", help="Use webcam")
    src.add_argument("--video-file", metavar="PATH", help="Path to a video file")
    src.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    # Features
    feat = p.add_argument_group("Features")
    feat.add_argument("--age", "--predict-age", action="store_true", dest="predict_age",
                       help="Enable age prediction")
    feat.add_argument("--gender", action="store_true", dest="predict_gender",
                       help="Enable gender prediction")
    feat.add_argument("--eyes", action="store_true", dest="detect_eyes",
                       help="Enable eye detection")
    feat.add_argument("--smile", action="store_true", dest="detect_smile",
                       help="Enable smile detection")
    feat.add_argument("--blur", action="store_true", dest="blur_faces",
                       help="Blur detected faces (privacy mode)")
    feat.add_argument("--dnn", action="store_true", dest="use_dnn",
                       help="Use DNN-based face detector instead of Haar")
    feat.add_argument("--all", action="store_true",
                       help="Enable all features (age, gender, eyes, smile)")

    # Tuning
    tune = p.add_argument_group("Detection parameters")
    tune.add_argument("--scale-factor", type=float, default=1.1,
                       help="Haar scale factor (default: 1.1)")
    tune.add_argument("--min-neighbors", type=int, default=5,
                       help="Haar min neighbors (default: 5)")
    tune.add_argument("--confidence", type=float, default=0.6,
                       help="DNN confidence threshold (default: 0.6)")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--output", "-o", metavar="PATH",
                      help="Output file path (image mode)")
    out.add_argument("--output-dir", metavar="DIR", default="output",
                      help="Directory for all outputs (default: output/)")
    out.add_argument("--no-display", action="store_true",
                      help="Don't display windows (headless mode)")

    # Misc
    misc = p.add_argument_group("Misc")
    misc.add_argument("--config", metavar="FILE", help="JSON config file")
    misc.add_argument("--download-models", action="store_true",
                       help="Download all required DNN models")
    misc.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Download models ──────────────────────────────────────────
    if args.download_models:
        download_models()
        print("Model download complete. You can now run detection.")
        return

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(args.config)

    # Override config with CLI flags
    if args.predict_age or args.all:
        cfg["predict_age"] = True
    if args.predict_gender or args.all:
        cfg["predict_gender"] = True
    if args.detect_eyes or args.all:
        cfg["detect_eyes"] = True
    if args.detect_smile or args.all:
        cfg["detect_smile"] = True
    if args.blur_faces:
        cfg["blur_faces"] = True
    if args.use_dnn:
        cfg["use_dnn"] = True
    cfg["scale_factor"] = args.scale_factor
    cfg["min_neighbors"] = args.min_neighbors
    cfg["dnn_confidence_threshold"] = args.confidence
    cfg["output_dir"] = args.output_dir

    # ── Create detector ──────────────────────────────────────────
    detector = FaceDetector(cfg)

    # ── Route to mode ────────────────────────────────────────────
    if args.image:
        result, infos = detector.detect_from_image(args.image)
        if result is not None:
            print(f"\nDetected {len(infos)} face(s)")
            for i, info in enumerate(infos, 1):
                parts = [f"Face {i}"]
                if "age" in info:
                    parts.append(f"Age: {info['age']}")
                if "gender" in info:
                    parts.append(f"Gender: {info['gender']}")
                if "eyes" in info:
                    parts.append(f"Eyes: {info['eyes']}")
                if "smiling" in info:
                    parts.append("Smiling" if info["smiling"] else "Not smiling")
                print("  " + "  |  ".join(parts))

            if not args.no_display:
                cv2.imshow("Face Detection Suite", result)
                print("\nPress any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    elif args.folder:
        detector.detect_from_folder(args.folder)

    elif args.video:
        print("Starting webcam detection...")
        print("Press 'h' for keyboard shortcuts, 'q' to quit\n")
        detector.start_video(args.camera)

    elif args.video_file:
        if not os.path.isfile(args.video_file):
            logger.error("Video file not found: %s", args.video_file)
            sys.exit(1)
        print(f"Processing video: {args.video_file}")
        print("Press 'h' for keyboard shortcuts, 'q' to quit\n")
        detector.start_video(args.video_file)

    else:
        parser.print_help()
        print("\nError: specify an input source (--image, --video, --video-file, or --folder)")
        sys.exit(1)


if __name__ == "__main__":
    main()
