import cv2
from utils import *


def newUserDataReceived(user: str):
    print(f"Welcome {user}")


lastUser = None
CAP = cv2.VideoCapture(0)
while CAP.isOpened():
    _, frame = CAP.read()
    output = detectAndValidateThenDecode(frame)
    if output is not None:
        data, lines = output
        if data != lastUser:
            lastUser = data
            newUserDataReceived(lastUser)
        for line in lines:
            cv2.line(frame, line[0], line[1], (0, 255, 0), 5)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
