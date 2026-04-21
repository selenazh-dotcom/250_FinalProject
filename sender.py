#sender

import cv2 as cv
import socket
import numpy as np

RPi_IP = "172.20.10.8"  # Replace with your RPi's IP
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    text = "TEST"
    font = cv.FONT_HERSHEY_SIMPLEX
    # cv.putText(frame, text, (10, 50), font, 1, (0, 255, 0), 2, cv.LINE_AA)


    
    cv.rectangle(frame, (5, 19), ((len(text)*25)+10, 60), 2, -1)
        
    cv.putText(frame, text, (10, 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    if ret:
        cv.imshow('Webcam',frame)
        
    if cv.waitKey(1) == ord('q'):
        break
    

    # Resize/compress to fit UDP packet limits (approx 65KB)
    _, encoded = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])
    data = encoded.tobytes()
    
    # Send in chunks if necessary, or ensure 'data' is < 65KB
    sock.sendto(data, (RPi_IP, PORT))

cap.release()
cv.destoryAllWindows()