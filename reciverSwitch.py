#recives a flag switch to flip functions

import cv2
import socket
import pickle
import numpy as np
import os
import pyttsx3


def vid():

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5235))
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
            
            cv2.imshow('Reciver Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    # out.release()#added
    # cv2.destoryAllWindows()

def speak():

    with open("words.txt", "r") as file:
        finalWords = file.read()
        pyttsx3.speak(finalWords)
        # cv2.destoryAllWindows()

def fromMiddle():
    print("Reciving video from the rpi")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5235))
    # text = "WORDS:\n"

    ##########################
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    root = os.getcwd()
    outpath=os.path.join(root,'cvSAVErpi.avi')
    out = cv2.VideoWriter(outpath,fourcc,20.0,(640,480))
    ##############################


    # with open("words.txt", "w") as file:
    #     file.write(text+" ")

    while True:
        data, addr = sock.recvfrom(65535) # Max UDP size
        obj = pickle.loads(data)
        if obj[0] == "FLAG": #added
            out.release()
            break


        elif obj[0] == "FRAME":
            frame = cv2.imdecode(np.frombuffer(obj[1], np.uint8), cv2.IMREAD_COLOR)
            # text = "HELLO WORLD"

            # font = cv2.FONT_HERSHEY_SIMPLEX


            # cv2.rectangle(frame, (5, 19), ((len(text)*25)+10, 60), 2, -1)
            
            # cv2.putText(frame, text, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)#added
            
            # cv2.imshow('Reciver Video', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

    


if __name__ == '__main__':
    vid()
    # speak()
    fromMiddle()