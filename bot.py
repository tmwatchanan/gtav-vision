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

from processing import detect_face, detect_head, detect_nav, get_cnn_image
from processing_yolo import YoloModel
from record_input import key_check, keys_to_output

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

def auto_drive(h, angle):
    if h is None or angle is None:
        return

    if angle > 5:
        PressKey(A)
        ReleaseKey(D)
    elif angle < -5:
        PressKey(D)
        ReleaseKey(A)
    else:
        ReleaseKey(A)
        ReleaseKey(D)

    if h > 0:
        PressKey(W)
    else:
        ReleaseKey(W)

def main():
    hwnd = win32gui.FindWindow(None, "FiveM - GTA FIVEM 1%")
    set_window_position(hwnd, top=False)

    if HEAD_DETECTION:
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
        # cv2.imwrite("ss.jpg", ss)
        img = ss.copy()
        img = cv2.resize(img, (OVERLAY_WIDTH, OVERLAY_HEIGHT))

        overlay_img = blank_img.copy()
        if FACE_DETECTION:
            overlay_img, (aim_x, aim_y) = detect_face(overlay_img, img)
        if HEAD_DETECTION:
            overlay_img = detect_head(yolo_model, img, overlay_img)
        if NAV_DETECTION:
            overlay_img, nav_th, h, angle = detect_nav(img, overlay_img)
            global auto_pilot
            if auto_pilot:
                auto_drive(h, angle)
            cnn_img = get_cnn_image(img, nav_th)


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
