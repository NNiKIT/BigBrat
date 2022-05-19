import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from skimage import io
from tqdm import tqdm

from myutils import crop_image


def get_edinburgh():
    """Creates dataset of chairs and chairs with person from Edinburgh university data."""
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5m, yolov5l, yolov5x, custom
    pathes = ["./day_1/", "./day_14/", "./day_17/", "./day_19/", "./day_20/"]  #
    for path in pathes:
        images = [i for i in os.listdir(path) if i.endswith(".jpg")]
        for img in tqdm(images):
            results = model(path + img)
            data = results.pandas().xyxy[0]
            t = data[data["name"].isin(["person", "chair"])].reset_index()
            crop_image(path + img, t)
        cl1 = len([i for i in os.listdir("./edinburgh/1/") if i.endswith(".jpg")])
        cl0 = len([i for i in os.listdir("./edinburgh/0/") if i.endswith(".jpg")])
        print("Class 1: {}. Class 0: {}".format(cl1, cl0))


if __name__ == "__main__":
    get_edinburgh()
