"""
model2.py

Repo: https://github.com/AkramOM606/American-Sign-Language-Detection

Architecture:
    - MediaPipe Hands extracts 21 landmarks per frame
    - Landmarks converted to wrist-relative, max-normalized 42-D vector (x,y only)
    - TFLite keypoint classifier predicts A-Z (single frame, no sequence needed)
    - SentenceBuilder accumulates confirmed letters into words/sentences

Files needed (copy from repo):
    model/keypoint_classifier/keypoint_classifier.tflite
    model/keypoint_classifier/keypoint_classifier_label.csv
    utils/cvfpscalc.py
"""

from __future__ import annotations

import os
import csv
import copy
import itertools
import collections
import numpy as np
import cv2
import mediapipe as mp

# ── Config ────────────────────────────────────────────────────────────────────

_DIR        = '/Users/selenazhang/Github/250_FinalProject/tmp_repo'
TFLITE_PATH = 'tmp_repo/model/keypoint_classifier/keypoint_classifier.tflite'
LABELS_PATH = '/Users/selenazhang/Github/250_FinalProject/tmp_repo/model/keypoint_classifier/keypoint_classifier_label.csv'
CONFIDENCE_THRESHOLD = 0.80   # minimum softmax score to display a prediction
CONFIRM_FRAMES       = 5     # consecutive stable frames before letter is committed


# ── TFLite keypoint classifier (matches repo's KeyPointClassifier class) ──────

class KeyPointClassifier:
    """
    Thin wrapper around the TFLite model from the AkramOM606 repo.
    Input:  42-D float32 vector  (21 landmarks × x,y — wrist-relative, normalized)
    Output: class index (int)
    """

    def __init__(self, model_path: str = TFLITE_PATH, num_threads: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TFLite model not found at '{model_path}'.\n"
                "Clone the repo and copy model/keypoint_classifier/ into your project:\n"
                "  https://github.com/AkramOM606/American-Sign-Language-Detection"
            )
        import tensorflow as tf
        self._interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )
        self._interpreter.allocate_tensors()
        self._input  = self._interpreter.get_input_details()
        self._output = self._interpreter.get_output_details()

    def __call__(self, landmark_list: list[float]) -> tuple[int, float]:
        """
        Args:
            landmark_list: 42 floats produced by pre_process_landmark()
        Returns:
            (class_index, confidence)
        """
        x = np.array([landmark_list], dtype=np.float32)
        self._interpreter.set_tensor(self._input[0]["index"], x)
        self._interpreter.invoke()
        result = self._interpreter.get_tensor(self._output[0]["index"])
        idx        = int(np.argmax(np.squeeze(result)))
        confidence = float(np.squeeze(result)[idx])
        return idx, confidence


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(path: str = LABELS_PATH) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label CSV not found at '{path}'.")
    with open(path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f)]


# ── Feature engineering (exact match to repo's pre_process_landmark) ──────────

def calc_landmark_list(image, landmarks) -> list[list[int]]:
    """Convert MediaPipe normalized landmarks → pixel coords."""
    h, w = image.shape[:2]
    points = []
    for lm in landmarks.landmark:
        points.append([
            min(int(lm.x * w), w - 1),
            min(int(lm.y * h), h - 1),
        ])
    return points


def pre_process_landmark(landmark_list: list[list[int]]) -> list[float]:
    """
    1. Make coordinates relative to wrist (landmark 0)
    2. Flatten to 1-D
    3. Divide by max absolute value  →  all values in [-1, 1]
    """
    tmp = copy.deepcopy(landmark_list)
    base_x, base_y = tmp[0]
    for pt in tmp:
        pt[0] -= base_x
        pt[1] -= base_y
    flat = list(itertools.chain.from_iterable(tmp))
    max_val = max(map(abs, flat))
    return [v / max_val for v in flat]   # 42-D


# ── Bounding box ──────────────────────────────────────────────────────────────

def calc_bounding_rect(image, landmarks) -> list[int]:
    h, w = image.shape[:2]
    pts = np.array([
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks.landmark
    ])
    x, y, bw, bh = cv2.boundingRect(pts)
    return [x, y, x + bw, y + bh]


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_landmarks(frame, landmark_list: list[list[int]]):
    """Draw the hand skeleton using pixel-coord landmark_list."""
    connections = [
        # Thumb
        (2,3),(3,4),
        # Index
        (5,6),(6,7),(7,8),
        # Middle
        (9,10),(10,11),(11,12),
        # Ring
        (13,14),(14,15),(15,16),
        # Pinky
        (17,18),(18,19),(19,20),
        # Palm
        (0,1),(1,2),(2,5),(5,9),(9,13),(13,17),(17,0),
    ]
    for s, e in connections:
        cv2.line(frame, tuple(landmark_list[s]), tuple(landmark_list[e]), (0,0,0),   6)
        cv2.line(frame, tuple(landmark_list[s]), tuple(landmark_list[e]), (255,255,255), 2)

    fingertips = {4, 8, 12, 16, 20}
    for i, pt in enumerate(landmark_list):
        r = 8 if i in fingertips else 5
        cv2.circle(frame, tuple(pt), r, (255,255,255), -1)
        cv2.circle(frame, tuple(pt), r, (0,0,0), 1)


def draw_bounding_rect(frame, brect: list[int]):
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0,0,0), 1)


def draw_info_text(frame, brect, handedness, label: str, confidence: float):
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[1]-22), (0,0,0), -1)
    hand_side = handedness.classification[0].label
    text = f"{hand_side}: {label} ({confidence*100:.0f}%)" if label else hand_side
    cv2.putText(frame, text, (brect[0]+5, brect[1]-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)


def draw_overlay(frame, fps: float, sentence_builder: "SentenceBuilder"):
    """Bottom bar showing running sentence + FPS."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-55), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, sentence_builder.full_text[-70:] or " ",
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"FPS:{fps:.0f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Q:quit  SPC:space  BKSP:delete  C:clear",
                (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)


# ── Sentence builder ──────────────────────────────────────────────────────────

class SentenceBuilder:
    """
    Commits a letter only when it's predicted stably for CONFIRM_FRAMES
    consecutive frames above CONFIDENCE_THRESHOLD.
    """

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES):
        self.confirm_frames  = confirm_frames
        self._buffer         = collections.deque(maxlen=confirm_frames)
        self.current_word    = ""
        self.sentence        = ""
        self._last_confirmed = ""

    def update(self, label: str, confidence: float) -> str | None:
        if confidence < CONFIDENCE_THRESHOLD:
            self._buffer.clear()
            return None

        self._buffer.append(label)
        if (len(self._buffer) == self.confirm_frames
                and len(set(self._buffer)) == 1
                and label != self._last_confirmed):

            self._last_confirmed = label
            self.current_word   += label
            self._buffer.clear()
            return label
        return None

    def add_space(self):
        self.sentence       += self.current_word + " "
        self.current_word    = ""
        self._last_confirmed = ""

    def backspace(self):
        if self.current_word:
            self.current_word    = self.current_word[:-1]
            self._last_confirmed = self.current_word[-1] if self.current_word else ""
        elif self.sentence:
            self.sentence     = self.sentence.rstrip()
            self.sentence, _, self.current_word = self.sentence.rpartition(" ")
            if self.sentence:
                self.sentence += " "

    def clear(self):
        self.current_word    = ""
        self.sentence        = ""
        self._last_confirmed = ""
        self._buffer.clear()

    @property
    def full_text(self) -> str:
        return self.sentence + self.current_word


# ── Convenience loader (called by main.py) ───────────────────────────────────

def load_model():
    """Returns (classifier, labels) ready for use in main.py."""
    classifier = KeyPointClassifier()
    labels     = load_labels()
    print(f"[model] Loaded TFLite classifier — {len(labels)} classes: {labels}")
    return classifier, labels