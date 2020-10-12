import math
from typing import Optional, Tuple, List, Final, Union
from functools import lru_cache
import qrcode
import cv2
import numpy as np
from PIL.Image import Image
from cv2 import QRCodeDetector

detector: QRCodeDetector = cv2.QRCodeDetector()


def generateQRCodeImage(data: str) -> np.array:
    img = qrcode.make(data).convert('RGB')
    # qr = qrcode.QRCode(version=3, box_size=10, border=4)
    # qr.add_data(str(data))
    # qr.make()
    # img = qr.make_image(fill_color="white", back_color="green").convert('RGB')
    return np.asarray(img)


def readQRCodeFromPoints(points):
    pass


def readQRCodeImage(image: Image) -> Optional[str]:
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dataDecoded, boxPoints, straight_qrcode = detector.detectAndDecode(img)
    if boxPoints is None: return None
    lines = extractLinesFromBoxPoints(boxPoints)
    if not isValidBoxLines(lines): return None
    return dataDecoded


def detectQRCodeImageBoxLines(image: Image) -> Union[None, list, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    _, boxPoints = detector.detect(image)
    if boxPoints is None: return None
    lines = extractLinesFromBoxPoints(boxPoints)
    if not isValidBoxLines(lines): return None
    return lines


def detectAndValidateThenDecode(image) -> Optional[Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]]:
    _, boxPoints = detector.detect(image)
    if boxPoints is None: return None
    lines = extractLinesFromBoxPoints(boxPoints)
    if lines is None: return None
    if not isValidBoxLines(tuple(lines)): return None
    data, _ = detector.decode(image, boxPoints)
    if data == "": return None
    return data, lines


@lru_cache()
def isValidBoxLines(lines: Tuple[Tuple[Tuple[int, int], Tuple[int, int]]]) -> bool:
    wrongMargin = 10
    ((x0, y0), (x1, y1)) = lines[0]
    firstLineLength = getDistanceBetween2Points(x0, y0, x1, y1)
    ((x0, y0), (x1, y1)) = lines[2]
    thirdLineLength = getDistanceBetween2Points(x0, y0, x1, y1)
    if abs(firstLineLength - thirdLineLength) > wrongMargin: return False
    ((x0, y0), (x1, y1)) = lines[1]
    secondLineLength = getDistanceBetween2Points(x0, y0, x1, y1)
    ((x0, y0), (x1, y1)) = lines[3]
    lastLineLength = getDistanceBetween2Points(x0, y0, x1, y1)
    if abs(secondLineLength - lastLineLength) > wrongMargin: return False
    return True


@lru_cache()
def getDistanceBetween2Points(x1: int, y1: int, x2: int, y2: int) -> float:
    distance = np.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))
    return distance


def extractLinesFromBoxPoints(boxPoints):
    outputLinesList: List[Tuple[Tuple[int, int], Tuple[int, int]]] = list()
    pointCount = len(boxPoints[0])
    if pointCount != 4: return None
    for i in range(pointCount):
        try:
            startPoint = int(boxPoints[0][i][0]), int(boxPoints[0][i][1])
            endPoint = int(boxPoints[0][(i + 1) % pointCount][0]), int(boxPoints[0][(i + 1) % pointCount][1])
            outputLinesList.append((startPoint, endPoint))
        except OverflowError:
            return None

    return outputLinesList
