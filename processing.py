import os

import cv2
import numpy as np

from config import Config

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

def detect_head(yolo_model, img, overlay_img):
    out_df, detected_img = yolo_model.detect(img)
    for index, row in out_df.iterrows():
        overlay_img = cv2.rectangle(overlay_img, (row.xmin, row.ymin), (row.xmax, row.ymax), (255, 255, 0), thickness=1)
    for index, row in out_df.iterrows():
        aim_x = int((row.xmin + row.xmax) / 2)
        aim_y = int((row.ymin + row.ymax) / 2)
        overlay_img = cv2.circle(overlay_img, (aim_x, aim_y), radius = 4, color=(0, 0, 255), thickness=2)
        break
    return overlay_img

def detect_nav(img, overlay_img):
    lower_nav_color = (160, 100, 170)
    upper_nav_color = (170, 255, 255)

    roi = img[Config.MAP_RANGE + tuple([slice(None)])]
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(roi_hsv, lower_nav_color, upper_nav_color)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    nav_roi = roi.copy()
    nav_roi[mask == 0] = 0

    th = cv2.cvtColor(nav_roi, cv2.COLOR_BGR2GRAY)
    th[th > np.max(th) / 2] = 255

    h, angle_with_y = find_nav_line(th, overlay_img, roi.shape)

        # cv2.imwrite("overlay_roi.jpg", overlay_roi)
        # cv2.imwrite("mask_roi.jpg", mask_roi)
    return overlay_img, th, h, angle_with_y

def find_nav_line(th, overlay_img, roi_shape):
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask_roi = np.zeros((roi_shape[0], roi_shape[1]), np.uint8)
    overlay_roi = np.zeros(roi_shape, np.uint8)

    h = None
    angle_with_y = None
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        cv2.drawContours(mask_roi, [cnt], 0, 255, -1)

        mask_roi = cv2.ximgproc.thinning(mask_roi,None,cv2.ximgproc.THINNING_GUOHALL)
        overlay_roi[mask_roi > 0] = (255, 0, 0)

        minLineLength = 10
        maxLineGap = 10
        lines = cv2.HoughLinesP(mask_roi,1,np.pi/180,10,np.array([]), minLineLength,maxLineGap)
        if lines is not None:
            dist = []
            for line in lines:
                x1,y1,x2,y2 = line[0]

                d1 = np.sum(np.abs(np.subtract((x1, y1), Config.ARROW_POS)))
                d2 = np.sum(np.abs(np.subtract((x2, y2), Config.ARROW_POS)))
                d = d1 + d2
                dist.append(d)

            m = np.argmin(dist)

            x1,y1,x2,y2 = lines[m][0]
            cv2.line(overlay_roi,(x1,y1),(x2,y2),(0,255,0),2)

            d1 = np.sum(np.subtract(Config.ARROW_POS, (x1, y1)))
            d2 = np.sum(np.subtract(Config.ARROW_POS, (x2, y2)))
            d = d1 + d2
            # + is above, - is beneath

            slope = (y2 - y1) / (x2 - x1)
            angle_with_y = np.arctan((x2 - x1) / (y2 - y1)) # radians
            angle_with_y = np.degrees(angle_with_y)

            h = np.max([y1, y2]) - np.min([y1, y2])

            # print(f"h={h}, angle={angle_with_y:.0f}")

        overlay_img[Config.MAP_RANGE + tuple([slice(None)])] = overlay_roi
    return h, angle_with_y

def get_cnn_gray_image(img, nav_th):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cnn_img = gray_img.copy()
    # place nav line
    cnn_img[Config.MAP_RANGE] = nav_th
    # place arrow
    arrow = gray_img[Config.ARROW_RANGE]
    _, arrow_th = cv2.threshold(arrow, 127, 255, cv2.THRESH_BINARY)
    cnn_img[Config.ARROW_RANGE] = arrow_th

    # move map to top right
    cnn_img[Config.TOP_RIGHT_MAP_RANGE] = cnn_img[Config.MAP_RANGE]
    # crop the bottom part out
    cnn_img = cnn_img[:Config.IGNORE_BOTTOM_HEIGHT, :]
    # cv2.imwrite("cnn_img.jpg", cnn_img)
    return cnn_img

def get_cnn_image(img, nav_th):
    cnn_img = img.copy()
    # place nav line
    cnn_img[Config.MAP_RANGE] = cv2.cvtColor(nav_th, cv2.COLOR_GRAY2RGB)
    # place arrow
    arrow = cv2.cvtColor(img[Config.ARROW_RANGE], cv2.COLOR_RGB2GRAY)
    _, arrow_th = cv2.threshold(arrow, 127, 255, cv2.THRESH_BINARY)
    cnn_img[Config.ARROW_RANGE] = cv2.cvtColor(arrow_th, cv2.COLOR_GRAY2RGB)

    # move map to top right
    cnn_img[Config.TOP_RIGHT_MAP_RANGE] = cnn_img[Config.MAP_RANGE]
    # crop the bottom part out
    cnn_img = cnn_img[:Config.IGNORE_BOTTOM_HEIGHT, :]
    cnn_img = cv2.resize(cnn_img, (Config.CNN_WIDTH, Config.CNN_HEIGHT))
    # cv2.imwrite("cnn_img-new.jpg", cnn_img)
    return cnn_img

if __name__ == "__main__":
    OVERLAY_WIDTH = 1024
    OVERLAY_HEIGHT = 576
    img = cv2.imread("screen-original_img.png")
    img = cv2.resize(img, (OVERLAY_WIDTH, OVERLAY_HEIGHT))
    cv2.imwrite("screen-img.png", img)
    height, width, _ = img.shape
    overlay_img = np.zeros((height, width, 3), np.uint8)
    overlay_img, nav_th, h, angle = detect_nav(img, overlay_img)
    cnn_img = get_cnn_image(img, nav_th)
    cv2.imwrite("screen-nav_th.png", nav_th)
    cv2.imwrite("screen-cnn_img.png", cnn_img)