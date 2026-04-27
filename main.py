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
    7. Send frame + translation to RPi server

Controls:
    Q          - quit
    C          - clear sentence


Claude was used to help understand the pipeline. 
It also assisted with the frame annotation functions.
    
"""

from __future__ import annotations

import time
import cv2
import mediapipe as mp
import numpy as np
import struct, socket
import math
import pickle


from capture import CameraCapture, landmarker
from model2 import (
    load_model,
    pre_process_landmark,
    draw_landmarks,
    draw_overlay,
    SentenceBuilder,
)
from utils.cvfpscalc import CvFpsCalc

# RPi Server Config

RPI_HOST    = "172.20.10.8"   
RPI_PORT    = 5005            
SEND_TO_RPI = True            # flip to True once RPi server is ready

# # Keep payload safely below UDP limits.
# MAX_DATAGRAM_SIZE = 1200

# # Header format:
# # frame_id (I), frame/translation (1), chunk_id (H), total_chunks (H), payload_len (H)
# HEADER_FMT = "!IBHHH"
# HEADER_SIZE = struct.calcsize(HEADER_FMT)
# MAX_CHUNK_PAYLOAD = MAX_DATAGRAM_SIZE - HEADER_SIZE

# MSG_TYPE_FRAME = 0
# MSG_TYPE_TEXT = 1


# Setup UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = (RPI_HOST, RPI_PORT)

_last_send_time = 0.0

def send_to_rpi(frame, translation: str, flag) -> None:
    global _last_send_time

    # the quit flag
    if flag == 1:
        for i in range(10):
            flg = pickle.dumps(["FLAG", flag])
            sock.sendto(flg, (RPI_HOST, RPI_PORT))
        return

    now = time.time()
    # limitting frame rate so socket isn't overwhelmed
    if now - _last_send_time < 0.05:   
        return
    _last_send_time = now

    # image frame
    small = cv2.resize(frame, (320, 240))
    _, buffer = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    frame_msg = pickle.dumps(["FRAME", buffer])

    print(f"[send] packet size: {len(frame_msg)} bytes")

    try:
        sock.sendto(frame_msg, target)
    # if packet size is too big or something, don't want entire thing to break
    except OSError as e:
        print(f"[send] frame dropped: {e}")
        return   # skip text too — wait for next frame

    # text translation
    try:
        msg = pickle.dumps(["TEXT", translation])
        sock.sendto(msg, target)
    except OSError as e:
        print(f"[send] text dropped: {e}")


# ~~~had help from CLAUDE with this section of functions~~~~~

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



def main() -> None:
    classifier, labels = load_model()
    cam      = CameraCapture(camera_index=0, preview=False)
    builder  = SentenceBuilder()
    fps_calc = CvFpsCalc(buffer_len=10)

    flagSwitch = 0

    cam.start()
    print("ASL Recognition running.")
    print("Controls: Q=quit | C=clear")

    try:
        while cam.running:

            fps = fps_calc.get()

            # get frame
            frame = cam.read()
            if frame is None:
                break

            # annotate
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

                # get features
                landmark_list = calc_landmark_list_tasks(frame, hand_landmarks)
                features      = pre_process_landmark(landmark_list)
                brect         = calc_bounding_rect_tasks(frame, hand_landmarks)

                # model prediction
                idx, confidence = classifier(features)
                label = labels[idx] if confidence >= 0.80 else ""

                # annotations
                if label:
                    builder.update(label, confidence)
                else:
                    builder._buffer.clear()

                draw_landmarks(frame, landmark_list)
                draw_bounding_rect_and_label(
                    frame, brect, handedness_label, label, confidence
                )

            else:
                builder._buffer.clear()

            draw_overlay(frame, fps, builder)
            cv2.imshow("ASL Recognition", frame)

            # sending to RPi
            if SEND_TO_RPI:
                send_to_rpi(frame, builder._last_confirmed, flagSwitch)
                if flagSwitch == 1:
                    break

            # keyboard controls to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                flagSwitch = 1
                # break
            # elif key == ord(" "):
            #     builder.add_space()
            # elif key == 8:               # backspace
            #     builder.backspace()
            elif key == ord("c"):
                builder.clear()

    finally:
        cam.stop()
        print("\nFinal text:", builder.full_text)


if __name__ == "__main__":
    main()
