
import os

from config import Config

def get_new_training_name():
    count = 1
    while True:
        data_filename = get_training_name(count)
        if os.path.isfile(data_filename):
            print('File exists, moving along',count)
            count += 1
        else:
            print('File does not exist, starting fresh!',count)
            return data_filename, count

def get_training_name(count):
    return Config.TRAINING_DATA_PATH.replace("X", str(count))