#this is the hub, with save everything and send everything at the end to avoid destroying live data
#recives a flag switch to flip functions

import cv2
import socket
import pickle
import numpy as np
import os
#import time
# import pyttsx3

LAP_A_PORT = 5005
LAP_B_IP = "172.20.10.7" 
LAP_B_PORT = 5235


def vid():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', LAP_A_PORT))
    text = "WORDS:\n"

    ##########################
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    root = os.getcwd()
    outpath=os.path.join(root,'cvSAVE.avi')
    out = cv2.VideoWriter(outpath,fourcc,20.0,(640,480))
    ##############################


    with open("words.txt", "w") as file:
        file.write(text+" ")

    while True:
        data, addr = sock.recvfrom(65535) # Max UDP size
        obj = pickle.loads(data)

        if obj[0] == "FLAG": #added
            for i in range(10):
                flg = pickle.dumps(["FLAG", 1])
                sock.sendto(flg, (LAP_B_IP, LAP_B_PORT))

            out.release()
            break
        
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

            # font = cv2.FONT_HERSHEY_SIMPLEX


            # cv2.rectangle(frame, (5, 19), ((len(text)*25)+10, 60), 2, -1)
            
            # cv2.putText(frame, text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)#added
            
            # cv2.imshow('Reciver Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        sock.sendto(data, (LAP_B_IP, LAP_B_PORT))

    # out.release()#added
    # cv2.destoryAllWindows()

# def speak():

#     with open("words.txt", "r") as file:
#         finalWords = file.read()
#         # pyttsx3.speak(finalWords)
#         # cv2.destoryAllWindows()

def fromMiddle():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("Now sending to the reciver")

    root = os.getcwd()
    vPath = os.path.join(root,'cvSave.avi')
    cap = cv2.VideoCapture(vPath)

    while cap.isOpened():
        ret,frame = cap.read()
        # cv2.imshow()
        # delay = int(1000,60)
        # if cv2.waitKey(delay) == ord(q)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_msg = pickle.dumps(["FRAME", buffer])
        sock.sendto(frame_msg, (LAP_B_IP, LAP_B_PORT))
    
    for i in range(10):
        flg = pickle.dumps(["FLAG", 1])
        sock.sendto(flg, (LAP_B_IP, LAP_B_PORT))

        # out.release()
        # break

    

    


    

if __name__ == '__main__':
    vid()
    # speak()
    #time.sleep(10)
    fromMiddle()
