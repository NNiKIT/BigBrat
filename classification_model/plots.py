import itertools
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_samples(X, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = 0  # int(n/j)
        c = 1
        plt.figure(figsize=(15, 6))
        for img in imgs:
            plt.subplot(1, j, c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.show()


def plot_loss_and_score(
    train_loss: List[float],
    val_loss: List[float],
    train_score: List[float],
    val_score: List[float],
    save_path: str = "",
) -> None:

    fg, ax = plt.subplots(1, 2, figsize=(19, 5))
    ax[0].plot(train_loss, label="train_loss")
    ax[0].plot(val_loss, label="val_loss")
    ax[0].set_title("Loss Curve")
    ax[0].legend(loc="best")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("loss")

    ax[1].plot(train_score, label="train_score")
    ax[1].plot(val_score, label="val_score")
    ax[1].set_title("Score Curve")
    ax[1].legend(loc="best")
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("score")
    plt.savefig(save_path)


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, save_path="./"
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path)
    # plt.show()
