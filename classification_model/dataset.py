import os
from typing import Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path_to_img: str, resize_shape: list = (256, 256), transform=None) -> None:
        self.img_dir = path_to_img

        self.images = []
        self.labels = []

        # class 1
        self.images += [
            self.img_dir + "1/" + i for i in os.listdir(self.img_dir + "1/") if i.endswith(".jpg")
        ]
        self.labels += [1] * len(self.images)

        # class 0
        self.images += [
            self.img_dir + "0/" + i for i in os.listdir(self.img_dir + "0/") if i.endswith(".jpg")
        ]
        self.labels += [0] * len(self.images)

        self.transform = transform
        self.resize_shape = tuple(resize_shape)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.array, int]:
        img_path = os.path.join(self.images[idx])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = cv2.resize(image, self.resize_shape)
        X = np.array(image, np.float32)
        X = np.rollaxis(X, 2, 0)
        Y = self.labels[idx]

        return X, Y
