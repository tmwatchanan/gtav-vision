import cv2
import numpy as np
import time

import win32gui
import win32ui
import win32con
import win32api
import winxpgui

from processing import detect_face

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

LEFT_OFFSET = 8
TOP_OFFSET = 32

def background_screenshot(hwnd, width, height, left=0, top=0):
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0), (width+left, height+top), dcObj, (left,top), win32con.SRCCOPY)

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

def setClickthrough(hwnd):
    try:
        styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        styles = win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
        win32gui.SetLayeredWindowAttributes(hwnd, 0, 255, win32con.LWA_ALPHA)
    except Exception as e:
        print(e)

def main():
    hwnd = win32gui.FindWindow(None, "FiveM - GTA FIVEM 1%")
    win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0) 
    first = True
    while True:
        last_time = time.time()

        img = background_screenshot(hwnd, WINDOW_WIDTH, WINDOW_HEIGHT, left=LEFT_OFFSET, top=TOP_OFFSET)
        # face_img = detect_face(screen)

        coords = (500, 500)
        img = cv2.circle(img, coords, radius=10, color=(255, 0, 0), thickness=2)

        cv_hwnd = display_image(img)
        if first:
            win32gui.SetWindowPos(cv_hwnd, win32con.HWND_TOPMOST, 0, 0, WINDOW_WIDTH+LEFT_OFFSET, WINDOW_HEIGHT+TOP_OFFSET, 0) 
            first = False
        win32gui.SetWindowLong(cv_hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong (hwnd, win32con.GWL_EXSTYLE ) | win32con.WS_EX_LAYERED )
        winxpgui.SetLayeredWindowAttributes(cv_hwnd, win32api.RGB(0,0,0), 180, win32con.LWA_ALPHA)
        setClickthrough(cv_hwnd)

        print(f'loop took {round(time.time()-last_time, 3)} seconds.')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()


# useful links
# https://stackoverflow.com/questions/41785831/how-to-optimize-conversion-from-pycbitmap-to-opencv-image