# 250_FinalProject
Alexandra Sierra - 4136096472

Selena Zhang - 2575286626


Mediapipe is a pretty old software, so it has very specific compatibility requirements.
Here are the libraries used:
-    python 3.11
-    numpy==1.24.3
-    opencv-python==4.8.1.78
-    mediapipe==0.10.x
-    tensorflow-macos==2.13.1
-    pickle
-    flask
-    pyttsx3


Installation Steps:

On Node 1 (sender):

- python -m pip install numpy==1.24.3
- python -m pip install opencv-python==4.8.1.78
- python -m pip install tensorflow-macos==2.13.1
- python -m pip install mediapipe
- wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task 
- git clone https://github.com/AkramOM606/American-Sign-Language-Detection.git

TO RUN:    python3 mainSwitch.py

On RPi:

- python -m pip install numpy==1.24.3
- python -m pip install opencv-python==4.8.1.78


TO RUN:    python3 middleSwitch.py

On Node 2 (receiver):

- python -m pip install opencv-python==4.8.1.78
- python -m pip install pyttsx3
- python -m pip install flask

TO RUN:    python3 receiver.py --port 5235 --web-port 8080








