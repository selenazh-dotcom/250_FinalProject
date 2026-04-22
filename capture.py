"""
capture.py

Camera capture from mac, with annotations
"""

from __future__ import annotations

import cv2
from typing import Optional
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

FINGERTIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

def draw_hand_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    for lm in hand_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)
        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


class CameraCapture:
    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        preview: bool = True,
        window_name: str = "Camera",
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.preview = preview
        self.window_name = window_name
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.running = True

    def read(self) -> Optional[any]:
        if not self.running or self.cap is None:
            raise RuntimeError("CameraCapture is not started. Call start() first.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        frame = cv2.flip(frame, 1)

        if self.preview:
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()
                return None
        return frame

    def annotate(self) -> Optional[any]:
        """
        Reads a frame, runs the MediaPipe Tasks hand landmarker on it,
        draws skeleton + fingertip labels, and returns the annotated frame.
        """
        frame = self.read()
        if frame is None:
            return None

        h, w, _ = frame.shape

        # Convert BGR → RGB and wrap in mediapipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in milliseconds (required for VIDEO mode)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand_landmarks in result.hand_landmarks:
            # Draw full skeleton
            draw_hand_landmarks(frame, hand_landmarks)

            # Label each fingertip
            for name, idx in FINGERTIP_INDICES.items():
                lm = hand_landmarks[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.putText(
                    frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                )
                cv2.circle(frame, (x, y), 7, (0, 255, 255), -1)

        cv2.imshow("Hand + Finger Tracking", frame)
        return frame

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.preview:
            cv2.destroyAllWindows()


def main() -> None:
    cam = CameraCapture(camera_index=0, preview=False)  # preview=False — annotate() handles imshow
    try:
        cam.start()
        print("Camera started. Press q to quit.")
        while cam.running:
            frame = cam.annotate()
            if frame is None:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cam.stop()
        print("Camera stopped.")


if __name__ == "__main__":
    main()