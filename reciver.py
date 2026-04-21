#reciver
import cv2 as cv
import socket
import numpy as np

PORT = 5006
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', PORT))

while True:
    packet, _ = sock.recvfrom(65536)
    data = np.frombuffer(packet, dtype=np.uint8)
    frame = cv.imdecode(data, cv.IMREAD_COLOR)

    
    
    if frame is not None:
        cv.imshow('Relayed Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
