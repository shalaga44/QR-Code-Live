import cv2
import numpy as np
from utils import *

text = "Eisra Osama"
img = generateQRCodeImage(text)
cv2.imwrite("eisra.png", img)
data = readQRCodeImage(img)
assert data == text
for line in detectQRCodeImageBoxLines(img):
    print(line)
    cv2.line(img, line[0], line[1], (0, 288, 0), 10)

cv2.imshow("img", img)
cv2.waitKey(0)
