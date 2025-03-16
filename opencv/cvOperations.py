import cv2
import numpy as np


class CVOperations:
    def __init__(self, message):
        self.message = message

    def grayScaleImage(self,src):
        gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        return gray_image

    def blurImage(self,src,ksize):
        self.src = src
        self.ksize = ksize

        self.dst = cv2.blur(src, ksize)
        return self.dst

    def blurImageNoParameters(self,src):
        self.src = src
        self.ksize = (6, 6)

        self.dst = cv2.blur(self.src, self.ksize)
        return self.dst

    def edgeDetectionNoParameters(self,src):
        self.src = src

        self.dst = cv2.Canny(self.src, 100, 200)
        return self.dst

    def edgeDetection(self, src, threshold1, threshold2, l2gradient):
        self.src = src
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.l2gradient = l2gradient

        self.dst = cv2.Canny(self.src, self.threshold1, self.threshold2, self.l2gradient)
        return self.dst

    def houghLines(self, src, rho, theta, threshold ):
        self.src = src
        self.rho = rho
        self.theta = theta
        self.threshold = threshold

        houghLines = cv2.HoughLines(self.src, self.rho, self.theta, self.threshold)
        return houghLines

    def houghLinesP(self, src, rho, theta, threshold ):
        self.src = src
        self.rho = rho
        self.theta = theta
        self.threshold = threshold

        houghLinesP = cv2.HoughLinesP(self.src, self.rho, self.theta, self.threshold)
        return houghLinesP