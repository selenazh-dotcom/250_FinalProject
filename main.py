"""
main.py

Entry point for the ASL sign-language recognition system.

Pipeline:
    1. Capture frame from webcam                      (capture.py)
    2. Run MediaPipe Tasks hand landmarker             (capture.py landmarker)
    3. Extract wrist-relative normalised features      (model.py)
    4. Predict ASL letter via TFLite classifier        (model.py)
    5. Accumulate confirmed letters into sentence      (model.py)
    6. Display annotated frame                         (opencv)
    7. [TODO] Send frame + translation to RPi server

Controls:
    Q          - quit
    SPACE      - insert space between words
    BACKSPACE  - delete last character
    C          - clear sentence
"""

from __future__ import annotations

import time
import cv2
import mediapipe as mp
import numpy as np
import struct, socket
import math

from capture import CameraCapture, landmarker
from model2 import (
    load_model,
    pre_process_landmark,
    draw_landmarks,
    draw_overlay,
    SentenceBuilder,
)
from utils.cvfpscalc import CvFpsCalc

# ── RPi Server Config (fill in when ready) ────────────────────────────────────

RPI_HOST    = "172.20.10.8"   # TODO: replace with your RPi's IP
RPI_PORT    = 5005             # TODO: match your RPi server port
SEND_TO_RPI = True            # flip to True once RPi server is ready

# Keep payload safely below UDP limits.
MAX_DATAGRAM_SIZE = 1200

# Header format:
# frame_id (I), frame/translation (1), chunk_id (H), total_chunks (H), payload_len (H)
HEADER_FMT = "!IBHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAX_CHUNK_PAYLOAD = MAX_DATAGRAM_SIZE - HEADER_SIZE

MSG_TYPE_FRAME = 0
MSG_TYPE_TEXT = 1


# ── RPi send stub ─────────────────────────────────────────────────────────────

def send_to_rpi(frame, translation: str, frame_id) -> None:
    """
    send to RPi via UDP

    Args:
        frame:          annotated BGR numpy array
        translation:    current sentence string from SentenceBuilder.full_text
    """
   
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # resizing so it can fit in chunks through UDP's size limit
    _, jpeg = cv2.imencode(".jpg", frame)
    jpeg_bytes = jpeg.tobytes()
    
    total_chunks = math.ceil(len(jpeg_bytes) / MAX_CHUNK_PAYLOAD)

    for chunk_id in range(total_chunks):
        start = chunk_id * MAX_CHUNK_PAYLOAD
        end = start + MAX_CHUNK_PAYLOAD
        payload = jpeg_bytes[start:end]

        header = struct.pack(
            HEADER_FMT,
            frame_id,
            MSG_TYPE_FRAME,
            chunk_id,
            total_chunks,
            len(payload),
        )

        packet = header + payload
        sock.sendto(packet, (RPI_HOST, RPI_PORT))

    # send translaiton
    if translation:
        text_bytes = translation.encode("utf-8")

        header = struct.pack(
            HEADER_FMT,
            frame_id,
            MSG_TYPE_TEXT,
            0,
            1,
            len(text_bytes),
        )
        sock.sendto(header + text_bytes, (RPI_HOST, RPI_PORT))

    pass


# ── Tasks API landmark helpers ────────────────────────────────────────────────
# The Tasks API returns landmarks as plain lists of objects with .x .y .z
# (not the old mp.solutions landmark proto), so we reimplement the two
# geometry helpers here rather than using the model.py versions which
# expected the old API's landmarks.landmark iterable.

def calc_landmark_list_tasks(frame, hand_landmarks) -> list[list[int]]:
    """Convert Tasks API hand_landmarks → pixel coords list."""
    h, w = frame.shape[:2]
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in hand_landmarks
    ]


def calc_bounding_rect_tasks(frame, hand_landmarks) -> list[int]:
    """Compute bounding rect from Tasks API hand_landmarks."""
    h, w = frame.shape[:2]
    pts = np.array([
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in hand_landmarks
    ])
    x, y, bw, bh = cv2.boundingRect(pts)
    return [x, y, x + bw, y + bh]


def draw_bounding_rect_and_label(frame, brect, hand_label, label, confidence):
    """Draw bounding box + prediction label above it."""
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    text = f"{hand_label}: {label} ({confidence*100:.0f}%)" if label else hand_label
    cv2.putText(frame, text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    classifier, labels = load_model()
    cam      = CameraCapture(camera_index=0, preview=False)
    builder  = SentenceBuilder()
    fps_calc = CvFpsCalc(buffer_len=10)

    frame_id = 0

    cam.start()
    print("ASL Recognition running.")
    print("Controls: Q=quit | SPACE=space | BACKSPACE=delete | C=clear")

    try:
        while cam.running:

            fps = fps_calc.get()

            # ── 1. Capture frame ──────────────────────────────────────
            frame = cam.read()
            if frame is None:
                break

            # ── 2. Tasks API landmark detection ───────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            label, confidence = "", 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]   # first hand
                handedness_label = (
                    result.handedness[0][0].display_name
                    if result.handedness else "Hand"
                )

                # ── 3. Extract features ───────────────────────────────
                landmark_list = calc_landmark_list_tasks(frame, hand_landmarks)
                features      = pre_process_landmark(landmark_list)
                brect         = calc_bounding_rect_tasks(frame, hand_landmarks)

                # ── 4. Predict letter ─────────────────────────────────
                idx, confidence = classifier(features)
                label = labels[idx] if confidence >= 0.80 else ""

                # ── 5. Update sentence ────────────────────────────────
                if label:
                    builder.update(label, confidence)
                else:
                    builder._buffer.clear()

                # ── Draw skeleton + bounding box + label ──────────────
                draw_landmarks(frame, landmark_list)
                draw_bounding_rect_and_label(
                    frame, brect, handedness_label, label, confidence
                )

            else:
                builder._buffer.clear()

            # ── 6. Draw overlay + show ────────────────────────────────
            draw_overlay(frame, fps, builder)
            cv2.imshow("ASL Recognition", frame)

            # ── 7. Send to RPi ────────────────────────────────────────
            if SEND_TO_RPI:
                send_to_rpi(frame, builder.full_text, frame_id)
                frame_id+=1

            # ── Keyboard controls ─────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                builder.add_space()
            elif key == 8:               # backspace
                builder.backspace()
            elif key == ord("c"):
                builder.clear()

    finally:
        cam.stop()
        print("\nFinal text:", builder.full_text)


if __name__ == "__main__":
    main()