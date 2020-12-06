import os
import time

import cv2
import keyboard
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
import winxpgui
import random

from processing import detect_face
from processing_yolo import YoloModel

from directkeys import PressKey, ReleaseKey, W, A, S, D

FACE_DETECTION = False
HEAD_DETECTION = False
NAV_DETECTION = True

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

OVERLAY_WIDTH = 1024
OVERLAY_HEIGHT = 576

WIDTH_OFFSET = 7
HEIGHT_OFFSET = 0
SS_LEFT_OFFSET = 2 + WIDTH_OFFSET
SS_TOP_OFFSET = 31 + HEIGHT_OFFSET
POS_LEFT_OFFSET = 10
POS_TOP_OFFSET = 39

ARROW_POS = (72, 72) # relative to ROI

DATASET_HEAD_DIR = os.path.join("dataset", "head")

def background_screenshot(hwnd, width, height, left=0, top=0):
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0), (width+left, height+top), dcObj, (left, top), win32con.SRCCOPY)

    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    # dataBitMap.SaveBitmapFile(cDC, 'screenshot.bmp')

    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

def display_image(img, width=None, height=None, title="DIP"):
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height))
    cv2.imshow(title, img)
    cv_hwnd = win32gui.FindWindow(None, title)
    return cv_hwnd

def overlay_cv_window(hwnd, alpha=100):
    try:
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), alpha, win32con.LWA_COLORKEY)
    except Exception as e:
        print(e)

def save_screenshot(img):
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    img_path = os.path.join(DATASET_HEAD_DIR, f"{timestr}.jpg")
    print(f"Saved {img_path}")
    cv2.imwrite(img_path, img)

def set_window_position(hwnd, top):
    topmost_con = win32con.HWND_TOPMOST if top else win32con.HWND_NOTOPMOST
    win32gui.SetWindowPos(hwnd, topmost_con, 0, 0, WINDOW_WIDTH + POS_LEFT_OFFSET, WINDOW_HEIGHT + POS_TOP_OFFSET, 0) 

running = True
def stop_running():
    global running
    running = False

auto_pilot = False

def set_auto_pilot(state):
    global auto_pilot
    auto_pilot = state
    print(f"auto_pilot {auto_pilot}")


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    #ReleaseKey(S)

def right():
    if random.randrange(0,3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():

    # if random.randrange(0,3) == 1:
    #     PressKey(W)
    # else:
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def main():
    hwnd = win32gui.FindWindow(None, "FiveM - GTA FIVEM 1%")
    set_window_position(hwnd, top=False)

    yolo_model = YoloModel()

    ss = None

    keyboard.on_press_key("=", lambda _: save_screenshot(ss))
    keyboard.on_press_key("|", lambda _: stop_running())
    keyboard.on_press_key("[", lambda _: set_auto_pilot(True))
    keyboard.on_press_key("]", lambda _: set_auto_pilot(False))

    blank_img = np.zeros((OVERLAY_HEIGHT, OVERLAY_WIDTH, 3), np.uint8)

    first = True
    start = time.time()
    frame_count = 0
    while True:
        last_time = time.time()

        ss = background_screenshot(hwnd, WINDOW_WIDTH, WINDOW_HEIGHT, left=SS_LEFT_OFFSET, top=SS_TOP_OFFSET)
        img = ss.copy()
        img = cv2.resize(img, (OVERLAY_WIDTH, OVERLAY_HEIGHT))

        overlay_img = blank_img.copy()
        # coords = (500, 500)
        # overlay_img = cv2.circle(overlay_img, coords, radius=10, color=(0, 0, 255), thickness=2)
        if FACE_DETECTION:
            overlay_img, (aim_x, aim_y) = detect_face(overlay_img, img)
        if HEAD_DETECTION:
            out_df, detected_img = yolo_model.detect(img)
            for index, row in out_df.iterrows():
                overlay_img = cv2.rectangle(overlay_img, (row.xmin, row.ymin), (row.xmax, row.ymax), (255, 255, 0), thickness=1)
            for index, row in out_df.iterrows():
                aim_x = int((row.xmin + row.xmax) / 2)
                aim_y = int((row.ymin + row.ymax) / 2)
                overlay_img = cv2.circle(overlay_img, (aim_x, aim_y), radius = 4, color=(0, 0, 255), thickness=2)
                break
        if NAV_DETECTION:
            lower_nav_color = (160, 160, 160)
            upper_nav_color = (170, 255, 255)

            lower_nav_head_color = (160, 170, 170)
            upper_nav_head_color = (170, 255, 255)

            # MAP_HEIGHT = (675, 834)
            # MAP_WIDTH = (49, 264)
            MAP_HEIGHT = (462, 556)
            MAP_WIDTH = (13, 158)
            # cv2.imwrite("img.jpg", img)
            # print(img.shape)
            roi = img[MAP_HEIGHT[0]:MAP_HEIGHT[1], MAP_WIDTH[0]:MAP_WIDTH[1] , :]
            cv2.imwrite("roi.jpg", roi)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(roi_hsv, lower_nav_color, upper_nav_color)
            head_mask = cv2.inRange(roi_hsv, lower_nav_head_color, upper_nav_head_color)
            kernel = np.ones((2,2), np.uint8)
            # head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_OPEN, kernel)
            head_mask = cv2.erode(head_mask, kernel, iterations = 1)
            head_mask = cv2.dilate(head_mask, kernel, iterations = 3)
            # cv2.imwrite("head_mask.jpg", head_mask)
            # overlay_img = cv2.bitwise_and(img, img, mask=mask)
            # overlay_img[mask > 0] = (255, 0, 0)
            # cv2.imwrite("nav.jpg", overlay_img)

            nav_roi = roi.copy()
            nav_roi[mask == 0] = 0
            nav_roi[head_mask > 0] = 0
# 
            gray = cv2.cvtColor(nav_roi, cv2.COLOR_BGR2GRAY)
            
            th = gray.copy()
            th[head_mask > 0] = 0
            th[th > np.max(th) / 2] = 255

            contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            mask_roi = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
            overlay_roi = np.zeros(roi.shape, np.uint8)

            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]

                cv2.drawContours(mask_roi, [cnt], 0, 255, -1)

                mask_roi = cv2.ximgproc.thinning(mask_roi,None,cv2.ximgproc.THINNING_GUOHALL)
                overlay_roi[mask_roi > 0] = (255, 0, 0)

                x,y,w,h = cv2.boundingRect(cnt)
                # cv2.rectangle(overlay_roi,(x,y),(x+w,y+h),(0,255,0),2)

                a = np.argwhere(mask_roi > 0)
                dist = np.sum(np.abs(a - ARROW_POS), axis=1)
                m = np.argmin(dist)
                nav_pos = a[m]

                minLineLength = 10
                maxLineGap = 10
                lines = cv2.HoughLinesP(mask_roi,1,np.pi/180,10,np.array([]), minLineLength,maxLineGap)
                if lines is not None:
                    for line in lines:
                        x1,y1,x2,y2 = line[0]
                        cv2.line(overlay_roi,(x1,y1),(x2,y2),(0,255,0),1)

                overlay_img[MAP_HEIGHT[0]: MAP_HEIGHT[1], MAP_WIDTH[0]:MAP_WIDTH[1], :] = overlay_roi

                cv2.imwrite("overlay_roi.jpg", overlay_roi)
                cv2.imwrite("mask_roi.jpg", mask_roi)


                global auto_pilot
                if auto_pilot:
                    if h > 20:
                        hwndChild = win32gui.GetWindow(hwnd, win32con.GW_CHILD)
                        temp = win32api.PostMessage(hwndChild, win32con.WM_CHAR, 'w', 0)
                        # straight()
                        PressKey(W)
                    else:
                        ReleaseKey(W)
                        
            # cv2.imshow("overlay_img", overlay_img)


        cv_hwnd = display_image(overlay_img, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        if first:
            set_window_position(cv_hwnd, top=True)
            first = False
        overlay_cv_window(cv_hwnd)

        # frame_count += 1
        # if frame_count % 120 == 0:
        #     fps = frame_count / (time.time() - start)
        #     print(f"FPS = {fps}")
        #     break

        # print(f'loop took {round(time.time()-last_time, 3)} seconds.')
        pressed_key = cv2.waitKey(1) & 0xFF
        if not running or pressed_key == ord('q'):
            break
    cv2.destroyAllWindows()
    if HEAD_DETECTION:
        del yolo_model



if __name__ == "__main__":
    main()


# useful links
# https://stackoverflow.com/questions/41785831/how-to-optimize-conversion-from-pycbitmap-to-opencv-image
# https://blog.insightdatascience.com/how-to-train-your-own-yolov3-detector-from-scratch-224d10e55de2
