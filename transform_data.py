import os

import cv2
import numpy as np
import pandas as pd

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

from config import Config
from data import get_training_name

TRAINING_PATH = "training"
TRAINING_IMAGE_PATH = os.path.join(TRAINING_PATH, "images")

def extract_npy():
    images = []
    labels = []
    for count in range(Config.TRANSFORM_START, Config.TRANSFORM_END + 1):
        data_filename = get_training_name(count)
        print(data_filename)
        data = np.load(data_filename)
        for i, (img, label) in enumerate(data, 1):
            img_name = data_filename.split("\\")[1]
            img_name = img_name.replace(".npy", f"-{i}.jpg")
            img_path = os.path.join(TRAINING_IMAGE_PATH, img_name)
            cv2.imwrite(img_path, img)
            images.append(img_name)
            labels.append(label)
    df_data = {
        "image": images,
        "label": labels
    }
    df = pd.DataFrame(df_data)
    print(df)
    csv_path = os.path.join(TRAINING_PATH, f"{Config.TRANSFORM_NAME}.csv")
    df.to_csv(csv_path, index=False)

def balance_data():
    csv_path = os.path.join(TRAINING_PATH, f"{Config.MODEL_NAME}.csv")
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts()
    print(counts)
    print(counts[1])
    df_w = df[df["label"] == "[1, 0, 0, 0, 0, 0, 0, 0, 0]"]
    df_w_sampled = df_w.sample(n=counts[1])
    print(df_w_sampled)
    df_new = df[df.index.isin(df_w_sampled.index) | ~df.index.isin(df_w.index)]
    print(df_new)

    new_csv_path = csv_path.replace(".csv", "-balanced.csv")
    df_new.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    # extract_npy()
    balance_data()
    # df = pd.read_csv(os.path.join(TRAINING_PATH, "gtav1.csv"))
    # a = df.iloc[0].label[1:-1].split(",")
