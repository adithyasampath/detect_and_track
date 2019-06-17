import keras
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
# import keras_retinanet
from keras_retinanet.models.resnet import ResNetBackbone
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import losses
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

labels_to_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


class Predictor:
    def __init__(self):
        self.num_classes = 80
        self.model_path = 'resnet50_coco_best_v2.1.0.h5'
        self.model = ResNetBackbone('resnet50').retinanet(self.num_classes)
        self.model.load_weights(
            self.model_path, by_name=True, skip_mismatch=True)
        self.model.compile(
            loss={
                'regression': losses.smooth_l1(),
                'classification': losses.focal()
            },
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))
        self.model = models.convert_model(self.model)
        print("Model Loaded and compiled")
        self.labels_to_names = labels_to_names

    def detect_image(self, frame):
        image = read_image_bgr(frame)
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict(
            np.expand_dims(image, axis=0))
        print("Boxes: ", boxes.shape)
        print("processing time: ", time.time() - start)
        # correct for image scale
        boxes /= scale
        boxes = tf.make_ndarray(boxes)
        scores = tf.make_ndarray(scores)
        labels = tf.make_ndarray(labels)
        return boxes, scores, labels

    def get_label(self, index):
        return self.labels_to_names[index]
