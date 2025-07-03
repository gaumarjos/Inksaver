# https://www.geeksforgeeks.org/python-thresholding-techniques-using-opencv-set-1-simple-thresholding/
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_path

PREFIX = "inksaved_"
SUFFIX = ""
BLACK_BACKGROUND_TH = 100.0
VERBOSE = False


class InkSaver:

    @staticmethod
    def generic(img, alpha, beta):
        return np.clip(img * alpha + beta, 0, 255)

    @staticmethod
    def gamma(img, gamma):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(img, lookUpTable)

    def convert_from_pdf(self):
        # Convert PDF pages to a list of PIL images
        pages = convert_from_path(self.input_filename)

        # A PDF can contain many pages, need to create one image and one output filename for each
        for i in range(len(pages)):
            # Convert first page to OpenCV image (NumPy array)
            img = np.array(pages[i])

            # Convert RGB (PIL) to BGR (OpenCV)
            self.img.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Prepare filenames
            self.output_filename.append(os.path.join(os.path.dirname(self.input_filename),
                                            "{}{}_p{}{}.jpg".format(PREFIX,
                                                                  os.path.basename(os.path.splitext(self.input_filename)[0]),
                                                                  i+1,
                                                                  SUFFIX)))

    def __init__(self, filename):
        self.input_filename = filename
        self.img = []
        self.output_filename = []

        # Check if it's an image or a PDF and load it accordingly
        if file.endswith((".pdf", ".PDF")):
            self.convert_from_pdf()
        else:
            self.img.append(cv2.imread(self.input_filename))
            self.output_filename.append(os.path.join(os.path.dirname(self.input_filename),
                                                     "{}{}{}.jpg".format(PREFIX, os.path.basename(
                                                         os.path.splitext(self.input_filename)[0]), SUFFIX)))

        self.processed_img = []

        if VERBOSE:
            print(self.output_filename)

    def process1(self):
        for img in self.img:
            # Convert the image to gray scale
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # Performing OTSU threshold
            ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

            thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 199, 5)

            thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 199, 5)

            self.processed_img.append(thresh1)

    def process2(self):
        for img in self.img:
            # Convert the image to gray scale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect if image is printed on black background and, in case, invert
            mean = np.mean(gray)
            if mean < BLACK_BACKGROUND_TH:
                gray = cv2.bitwise_not(gray)

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

            self.processed_img.append(gamma)

    def save(self):
        for processed_img, output_filename in zip(self.processed_img, self.output_filename):
            cv2.imwrite(output_filename, processed_img)

    def show(self):
        cv2.imshow('Original', self.img)
        cv2.imshow('Processed', self.processed_img)


def wrapper(fn):
    image = InkSaver(fn)
    image.process2()
    image.save()


if __name__ == '__main__':

    # Is it a folder?
    if os.path.isdir(sys.argv[1]):
        for folder, subfolders, files in os.walk(sys.argv[1]):
            for file in files:
                if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".pdf", ".PDF")):
                    if not file.endswith("{}.jpg".format(SUFFIX)):
                        filename = os.path.join(os.path.abspath(folder), file)
                        if VERBOSE:
                            print("Processing {}".format(filename))
                        wrapper(filename)

    # Assume they're files
    else:
        for file in sys.argv[1:]:
            if os.path.isfile(file):
                wrapper(file)
