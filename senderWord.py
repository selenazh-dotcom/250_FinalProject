
import cv2
import socket
import pickle

# Setup UDP Socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = ('10.0.2.15', 5005)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
   


    msg = pickle.dumps(["TEXT", "test test"])
    sock.sendto(msg, target)
    
    
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    frame_msg = pickle.dumps(["FRAME", buffer])
    sock.sendto(frame_msg, target)