# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

import os
import sys
import cv2
import numpy as np

SUFFIX = "inksaved"


class InkSaver():

    @staticmethod
    def generic(img, alpha, beta):
        return np.clip(img * alpha + beta, 0, 255)

    @staticmethod
    def gamma(img, gamma):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(img, lookUpTable)

    def __init__(self, filename):
        self.img = cv2.imread(filename)
        self.processed_img = []
        self.output_filename = filename[:-4] + "_{}.jpg".format(SUFFIX)

    def process1(self):
        print("Processing...")

        # Convert the image to gray scale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 199, 5)

        thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 199, 5)

        self.processed_img = thresh1

    def process2(self):
        # Convert the image to gray scale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Isolate, blur and subtract the background
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        blurred = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(gray, blurred)

        # Normalise image
        norm_img = diff_img.copy()  # Needed for 3.x compatibility
        cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Remove grayish shades
        _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)

        # Normalise again
        norm2_img = thr_img.copy()  # Needed for 3.x compatibility
        cv2.normalize(thr_img, norm2_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Gamma correction
        gamma = self.gamma(norm2_img, 4.0)

        self.processed_img = gamma

    def save(self):
        cv2.imwrite(self.output_filename, self.processed_img)

    def show(self):
        cv2.imshow('Original', self.img)
        cv2.imshow('Processed', self.processed_img)


def wrapper(filename):
    image = InkSaver(filename)
    image.process2()
    image.save()


if __name__ == '__main__':
    for arg in sys.argv:
        print(arg)

    path = sys.argv[len(sys.argv) - 1]
    if os.path.isdir(path):
        for folder, subfolders, files in os.walk(path):
            for file in files:
                if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG")):
                    if not file.endswith("{}.jpg".format(SUFFIX)):
                        filename = os.path.join(os.path.abspath(folder), file)
                        print("Processing {}".format(filename))
                        wrapper(filename)

    elif os.path.isfile(path):
        wrapper(path)
    else:
        print("Argument must be an existing directory or file.")
