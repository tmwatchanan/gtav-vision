import os
import sys
import cv2


src_path = "yolo"

sys.path.append(src_path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join("dataset")

image_folder = os.path.join(data_folder, "head")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

class YoloModel:
    def __init__(self):
        min_confidence = 0.25
        is_tiny = False

        if is_tiny and anchors_path:
            anchors_path = os.path.join(
                os.path.dirname(anchors_path), "yolo-tiny_anchors.txt"
            )

        anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")
        anchors = get_anchors(anchors_path)
        # define YOLO detector
        self.yolo = YOLO(
            **{
                "model_path": model_weights,
                "anchors_path": anchors_path,
                "classes_path": model_classes,
                "score": min_confidence,
                "gpu_num": 0,
                "model_image_size": (416, 416),
            }
        )

        # labels to draw on images
        class_file = open(model_classes, "r")
        self.input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    
    def __del__(self):
        # Close the current yolo session
        self.yolo.close_session()

    
    def detect(self, img, show_time=True):
        start = timer()
        prediction, detected_img = self.yolo.detect_image(img, show_stats=False)
        detected_img = np.asarray(detected_img)
        y_size, x_size, _ = detected_img.shape

        # Make a dataframe for the prediction outputs
        out_df = pd.DataFrame(
            columns=[
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label",
                "confidence",
                "x_size",
                "y_size",
            ]
        )

        for single_prediction in prediction:
            out_df = out_df.append(
                pd.DataFrame(
                    [
                        single_prediction
                        + [x_size, y_size]
                    ],
                    columns=[
                        "xmin",
                        "ymin",
                        "xmax",
                        "ymax",
                        "label",
                        "confidence",
                        "x_size",
                        "y_size",
                    ],
                )
            )
        end = timer()
        if show_time:
            print(f"Yolo v3 detection took {end-start:.2f} s")
        return out_df, detected_img


if __name__ == "__main__":
    yolo_model = YoloModel()
    img = cv2.imread("1.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out_df, detected_img = yolo_model.detect(img)
    cv2.imshow("yolo", detected_img)
    print(out_df)
    while True:
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            cv2.destroyAllWindows()
            break

