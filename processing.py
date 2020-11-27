import os

import cv2

ASSET_PATH = "assets"

def detect_face(overlay_img, img):
    # load the trained face cascade from opencv
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face_cascade = cv2.CascadeClassifier(os.path.join(ASSET_PATH, "haarcascade_frontalface_default.xml"))

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.05, 8)

    aim_x, aim_y = None, None
    for (x, y, w, h) in faces:
        cv2.rectangle(overlay_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        aim_x = (x + (x+w)) / 2
        aim_y = (y + (y+h)) / 2
        # break
    
    return overlay_img, (aim_x, aim_y)