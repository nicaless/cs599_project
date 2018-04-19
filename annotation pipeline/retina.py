import keras
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import sys

from PIL import Image
import glob

import pandas as pd
import tensorflow as tf

video_folder = sys.argv[1]


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

model_path = 'resnet50_coco_best_v1.2.2.h5'

model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}

folder_list = glob.glob(video_folder + "/*/")
number_folders = len(folder_list)


def bound(x):
    image = read_image_bgr(x)

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    # print("processing time: ", time.time() - start)

    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
    detections[0, :, :4] /= scale
    i = 0
    j = 0
    x = []
    y = []
    z = []
    h = []
    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):

        if score < 0.35:
            continue
        b = detections[0, idx, :4].astype(int)
        j = j + 1
        # print label
        if (label == 2 or label == 5 or label == 7):
            i = i + 1
            x.append(int(b[0]))
            y.append(int(b[1]))
            z.append(int(b[2] - b[0]))
            h.append(int(b[3] - b[1]))

    return x, y, z, h, i


for l in range(0, number_folders):
    top = []
    bottom = []
    width = []
    height = []
    density = []
    frame = []
    file_name = []
    n = 1
    # read images from folder
    for file in sorted(glob.glob(folder_list[l] + '*.png')):
        x, y, z, h, i = bound(file)  # get bounding boxes
        # print("frame number=",n)
        x = ",".join(str(bit) for bit in x)
        y = ",".join(str(bit) for bit in y)
        z = ",".join(str(bit) for bit in z)
        h = ",".join(str(bit) for bit in h)
        top.append(x)
        bottom.append(y)
        width.append(z)
        height.append(h)
        density.append(i)
        # frame.append(n)
        na = os.path.splitext(os.path.basename(file))[0]
        file_name.append(na)
        n = n + 1

# convert to pandas dataframes
d1 = pd.DataFrame({'top': top})

# df2=pd.Series(bottom)
d2 = pd.DataFrame({'bottom': bottom})

# df3=pd.Series(width)
d3 = pd.DataFrame({'width': width})

# df4=pd.Series(height)
d4 = pd.DataFrame({'height': height})

# df5=pd.Series(density)
d5 = pd.DataFrame({'car density': density})

# df6=pd.Series(frame)
d6 = pd.DataFrame({'frame number': file_name})

# d7=pd.DataFrame({'file name':file_name})

fin = pd.concat([d1, d2, d3, d4, d5, d6], axis=1)

csv_name = folder_list[l] + '*.png'
# dataframe to csv file
csv_file = "csv/" + os.path.basename(os.path.dirname(csv_name)) + ".csv"
fin.to_csv(csv_file)

