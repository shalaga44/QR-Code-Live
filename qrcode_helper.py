from functools import lru_cache
from typing import *

import cv2
import numpy
import numpy as np

# Type alias
OpenCVImage = numpy.ndarray
QrCodeData = str
BoxLinesHashableType = Tuple[Tuple[Tuple[int, int], Tuple[int, int]]]
BoxLinesType = List[Tuple[Tuple[int, int], Tuple[int, int]]]


class QrCodeHelper:
    def __init__(self):
        self.lastUser = None
        self.DETECTOR: Final = cv2.QRCodeDetector()
        self.CAMERA: Final = cv2.VideoCapture(0)

    def main(self):
        while self.CAMERA.isOpened():
            image = self._getNextFrameFromCamera()

            output = self.detectAndValidateThenDecode(image)
            if output is None: self._showImage(image); continue

            data, lines = output
            if self._isNewUser(data):
                self.registerNewUser(data)
            self._drawLines(image, lines)

            self._showImage(image)

    @staticmethod
    def _showImage(image: OpenCVImage) -> NoReturn:
        cv2.imshow("image", image)
        cv2.waitKey(1)

    def _getNextFrameFromCamera(self) -> OpenCVImage:
        _, image = self.CAMERA.read()
        return image

    def registerNewUser(self, data: str) -> NoReturn:
        self.lastUser = data
        print(self.lastUser)

    def _isNewUser(self, data: str) -> bool:
        return data != self.lastUser

    def detectAndValidateThenDecode(self, image: OpenCVImage) -> Optional[Tuple[QrCodeData, BoxLinesHashableType]]:
        _, boxPoints = self.DETECTOR.detect(image)
        if boxPoints is None: return None
        lines = QrCodeHelper.extractLinesFromBoxPoints(boxPoints)
        if lines is None: return None
        if not QrCodeHelper.isValidBoxLines(tuple(lines)): return None
        data, _ = self.DETECTOR.decode(image, boxPoints)
        if data == "": return None
        return str(data), lines

    @staticmethod
    @lru_cache()
    def isValidBoxLines(lines: BoxLinesHashableType) -> bool:
        wrongMargin = 10
        ((x0, y0), (x1, y1)) = lines[0]
        firstLineLength = QrCodeHelper.getDistanceBetween2Points(x0, y0, x1, y1)
        ((x0, y0), (x1, y1)) = lines[2]
        thirdLineLength = QrCodeHelper.getDistanceBetween2Points(x0, y0, x1, y1)
        if abs(firstLineLength - thirdLineLength) > wrongMargin: return False
        ((x0, y0), (x1, y1)) = lines[1]
        secondLineLength = QrCodeHelper.getDistanceBetween2Points(x0, y0, x1, y1)
        ((x0, y0), (x1, y1)) = lines[3]
        lastLineLength = QrCodeHelper.getDistanceBetween2Points(x0, y0, x1, y1)
        if abs(secondLineLength - lastLineLength) > wrongMargin: return False
        return True

    @staticmethod
    @lru_cache()
    def getDistanceBetween2Points(x1: int, y1: int, x0: int, y0: int) -> float:
        distance = np.sqrt(np.math.pow((x1 - x0), 2) + np.math.pow((y1 - y0), 2))
        return distance

    @staticmethod
    def extractLinesFromBoxPoints(boxPoints) -> Optional[BoxLinesHashableType]:
        outputLinesList: BoxLinesType = list()
        pointCount = len(boxPoints[0])
        if pointCount != 4: return None
        for i in range(pointCount):
            try:
                startPoint = int(boxPoints[0][i][0]), int(boxPoints[0][i][1])
                endPoint = int(boxPoints[0][(i + 1) % pointCount][0]), int(boxPoints[0][(i + 1) % pointCount][1])
                outputLinesList.append((startPoint, endPoint))
            except OverflowError:
                return None

        return tuple(outputLinesList)

    @staticmethod
    def _drawLines(image: OpenCVImage, lines: BoxLinesHashableType):
        for line in lines:
            cv2.line(image, line[0], line[1], (0, 255, 0), 5)


x = QrCodeHelper()
x.main()
