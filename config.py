import os

class Config:
    MAP_HEIGHT = (462, 556)
    MAP_WIDTH = (13, 158)
    MAP_RANGE = tuple([slice(MAP_HEIGHT[0], MAP_HEIGHT[1]), slice(MAP_WIDTH[0], MAP_WIDTH[1])])
    
    ARROW_POS = (72, 72) # relative to ROI
    
    ARROW_HEIGHT = (530, 541)
    ARROW_WIDTH = (80, 91)
    ARROW_RANGE = tuple([slice(ARROW_HEIGHT[0], ARROW_HEIGHT[1]), slice(ARROW_WIDTH[0], ARROW_WIDTH[1])])

    TOP_RIGHT_MAP_HEIGHT = (0, 94)
    TOP_RIGHT_MAP_WIDTH = (878, 1023)
    TOP_RIGHT_MAP_RANGE = tuple([slice(TOP_RIGHT_MAP_HEIGHT[0], TOP_RIGHT_MAP_HEIGHT[1]), slice(TOP_RIGHT_MAP_WIDTH[0], TOP_RIGHT_MAP_WIDTH[1])])

    IGNORE_BOTTOM_HEIGHT = 405

    TRAINING_DATA_PATH = os.path.join("data", "training_data-X.npy")


if __name__ == "__main__":
    import cv2

    print(Config.ARROW_RANGE)
    print(Config.MAP_RANGE)

    img = cv2.imread("img2.png")
    a = img[Config.ARROW_RANGE]
    cv2.imshow("a", a)
    cv2.waitKey()
