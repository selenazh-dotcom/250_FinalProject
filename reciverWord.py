
import cv2
import socket
import pickle
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5005))
text = "no data yet"
with open("words.txt", "w") as file:
    file.write(text+" ")

while True:
    data, addr = sock.recvfrom(65535) # Max UDP size
    obj = pickle.loads(data)
    
    if obj[0] == "TEXT":

        
        if(text!=obj[1] ):
            text =obj[1]
            # print(f"Message: {obj[1]}")
            print(f"Message: {text}")
            with open("words.txt", "a") as file:
                file.write(text+" ")



    elif obj[0] == "FRAME":
        frame = cv2.imdecode(np.frombuffer(obj[1], np.uint8), cv2.IMREAD_COLOR)
        # text = "HELLO WORLD"

        font = cv2.FONT_HERSHEY_SIMPLEX


        cv2.rectangle(frame, (5, 19), ((len(text)*25)+10, 60), 2, -1)
        
        cv2.putText(frame, text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Reciver Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break