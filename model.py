from tensorflow.keras import Model
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from config import Config


def model_inceptionv3():
    backbone = inception_v3.InceptionV3(
        include_top=False,
        weights=None, #"imagenet"
        input_tensor=None,
        input_shape=(Config.CNN_HEIGHT, Config.CNN_WIDTH, 3),
        pooling=None,
        classes=Config.NUM_CLASS,
        classifier_activation="softmax",
    )

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(Config.NUM_CLASS, activation='softmax')(x)
    model = Model(inputs=backbone.input, outputs=predictions)
    return backbone, model
