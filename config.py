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

    CNN_WIDTH = 500
    CNN_HEIGHT = 196

    TRAIN_RATIO = .8
    VALIDATION_RATIO = 1 - TRAIN_RATIO

    MODEL_NAME = "gtav3-500x196-highway-03-8.522"
    NUM_CLASS = 3

    LOAD_WEIGHTS_PATH = None # os.path.join("models", "gtav3-500x196-balanced-scratch-12-1.721.hdf5")
    INITIAL_EPOCH = 0 # start from 0
    TRAIN_EPOCH = 100
    TRAIN_START = 1
    TRAIN_END = 51
    LEARNING_RATE = 0.001
    DECAY_STEPS = 5
    DECAY_RATE = 0.9
    BATCH_SIZE = 40

    TRAINING_DATA_PATH = os.path.join("data", "training_data-X.npy")
    TRAINING_CSV_PATH = os.path.join("training", f"{MODEL_NAME}.csv")
    TRAINING_IMAGES_PATH = os.path.join("training", "images")

    TRANSFORM_NAME = "gtav3-500x196-highway"
    TRANSFORM_START = 1
    TRANSFORM_END = 89


if __name__ == "__main__":
    import cv2

    print(Config.ARROW_RANGE)
    print(Config.MAP_RANGE)

    img = cv2.imread("img2.png")
    a = img[Config.ARROW_RANGE]
    cv2.imshow("a", a)
    cv2.waitKey()
