import cv2
import time

import d3dshot

from processing import detect_face


WIDTH = 1280
HEIGHT = 720

GAME_WIDTH = 1280
GAME_HEIGHT = 720

def display_image(img):
    resized_img = cv2.resize(img, (WIDTH, HEIGHT))
    cv2.imshow('DIP', resized_img)


def main():
    d = d3dshot.create(capture_output="numpy")
    while True:
        screen = d.screenshot(region=(0, 30, GAME_WIDTH, GAME_HEIGHT))

        # screen = grab_screen(region=(0,40,GAME_WIDTH,GAME_HEIGHT+40))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        last_time = time.time()
        
        face_img = detect_face(screen)

        display_image(face_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        # print(f'loop took {round(time.time()-last_time, 3)} seconds.')


if __name__ == "__main__":
    main()