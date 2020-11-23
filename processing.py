import os

import cv2

ASSET_PATH = "assets"

def detect_face(img):
    # load the trained face cascade from opencv
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(ASSET_PATH, "haarcascade_frontalface_default.xml"))

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return img
