import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
from skimage import io


def plot(image, title=None, figsize=(10, 10), cmap="gray"):
    """Plot one image"""
    plt.rcParams["figure.figsize"] = (15, 15)
    plt.title(title)
    plt.imshow(image, cmap)
    # plt.show()


def show(img_path, data):
    """Plot bounding boxes of chairs and chairs with person on the image."""
    res = data
    image = io.imread(img_path)
    color = (255, 0, 0)
    for i in range(res.shape[0]):
        if res["name"][i] == "chair":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        pt1, pt2 = (int(res["xmin"][i]), int(res["ymin"][i])), (
            int(res["xmax"][i]),
            int(res["ymax"][i]),
        )
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.rectangle(image, (pt1[0], pt1[1] - 35), (pt2[0], pt1[1]), color, cv2.FILLED)
        cv2.putText(
            image,
            "{} {}".format(res["name"][i], str(round(res["confidence"][i], 2))),
            (int(res["xmin"][i]), int(res["ymin"][i])),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    plot(image)


def get_result(img_path, dataframe):
    """It is used to find chairs and chairs with person on one image. Result is the image with drawn rectangles."""
    image = io.imread(img_path)
    persons = dataframe[dataframe["name"] == "person"].reset_index()
    chairs = dataframe[dataframe["name"] == "chair"].reset_index()
    pairs = []
    for i in range(persons.shape[0]):
        person_center = (
            persons["xmin"][i] + 0.5 * (persons["xmax"][i] - persons["xmin"][i]),
            persons["ymin"][i] + 0.5 * (persons["ymax"][i] - persons["ymin"][i]),
        )
        min_euc = float("inf")
        pair = None
        for j in range(chairs.shape[0]):
            chair_center = (
                chairs["xmin"][j] + 0.5 * (chairs["xmax"][j] - chairs["xmin"][j]),
                chairs["ymin"][j] + 0.5 * (chairs["ymax"][j] - chairs["ymin"][j]),
            )
            euc_dist = distance.euclidean(person_center, chair_center)
            if euc_dist < 180 and euc_dist < min_euc:
                min_euc = euc_dist
                pair = (i, j)
        pairs.append(pair)
    for i, j in pairs:
        p = persons.iloc[i]
        c = chairs.iloc[j]
        xmin = int(min(p["xmin"], c["xmin"]))
        ymin = int(min(p["ymin"], c["ymin"]))
        xmax = int(max(p["xmax"], c["xmax"]))
        ymax = int(max(p["ymax"], c["ymax"]))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
    plot(image)


def crop_image(img_path, dataframe):
    """
    This function allows to crop and save chairs and chairs with person. It was used to create dataset from Edinburgh university images.
    It takes img_path and returned datafrane from yolo detection.
    """

    image = io.imread(img_path)
    persons = dataframe[dataframe["name"] == "person"].reset_index()
    chairs = dataframe[dataframe["name"] == "chair"].reset_index()
    pairs = []
    empty_chairs = []
    for i in range(chairs.shape[0]):
        chair_center = (
            chairs["xmin"][i] + 0.5 * (chairs["xmax"][i] - chairs["xmin"][i]),
            chairs["ymin"][i] + 0.5 * (chairs["ymax"][i] - chairs["ymin"][i]),
        )
        min_euc = float("inf")
        pair = None
        for j in range(persons.shape[0]):
            person_center = (
                persons["xmin"][j] + 0.5 * (persons["xmax"][j] - persons["xmin"][j]),
                persons["ymin"][j] + 0.5 * (persons["ymax"][j] - persons["ymin"][j]),
            )
            euc_dist = distance.euclidean(person_center, chair_center)
            if euc_dist < 180 and euc_dist < min_euc:
                min_euc = euc_dist
                pair = (i, j)
        if pair:
            pairs.append(pair)
        else:
            empty_chairs.append(i)
    for idx, pair in enumerate(pairs):
        i, j = pair
        c = chairs.iloc[i]
        p = persons.iloc[j]
        xmin = int(min(p["xmin"], c["xmin"]))
        ymin = int(min(p["ymin"], c["ymin"]))
        xmax = int(max(p["xmax"], c["xmax"]))
        ymax = int(max(p["ymax"], c["ymax"]))
        img = image[ymin:ymax, xmin:xmax]
        io.imsave("./edinburgh/1/{}_{}_1.jpg".format(img_path.split("/")[-1], idx), img)
    for idx, i in enumerate(empty_chairs):
        c = chairs.iloc[i]
        xmin = int(c["xmin"])
        ymin = int(c["ymin"])
        xmax = int(c["xmax"])
        ymax = int(c["ymax"])
        img = image[ymin:ymax, xmin:xmax]
        io.imsave("./edinburgh/0/{}_{}_0.jpg".format(img_path.split("/")[-1], idx), img)
