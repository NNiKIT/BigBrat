import os
import warnings

import cv2
from skimage.io import imread, imsave
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_images(path_to_video="./office/"):
    """It was used to save image from Office video streams each 30 seconds."""
    videos = [i for i in os.listdir(path_to_video) if i.endswith(".mp4")]

    path_to_save = "./images/"
    # os.mkdir('images')
    for video in videos:
        vidcap = cv2.VideoCapture(path_to_video + video)
        count = 0
        success = True
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))

        os.mkdir(path_to_save + video.split(".")[0])

        while True:
            success, image = vidcap.read()
            if success:
                if count % (30 * fps) == 0:
                    cv2.imwrite(
                        path_to_save + video.split(".")[0] + "/" + "frame%d.jpg" % count, image
                    )
                    print("successfully written 30th frame")
                count += 1
            else:
                break
    return path_to_save


def get_data(path_to_video, first_room_coords, second_room_coords, new_path):
    """Uses to create dataset from Office data (more details in README.md). It crops and save images according to 'left.txt' and 'right.txt' that store coordinates."""
    path = get_images(path_to_video)
    folders = os.listdir(path)

    # seat places coordinates of room 1
    left_bb = []
    with open(first_room) as f:
        for line in f:
            left_bb.append([int(el) for el in line.split(",")])
    # seat places coordinates of room 2
    right_bb = []
    with open(second_room) as f:
        for line in f:
            right_bb.append([int(el) for el in line.split(",")])

    cur_bb = None
    for folder in tqdm(folders):
        if "left" in folder:
            cur_bb = left_bb
        else:
            cur_bb = right_bb
        cur_path = path + folder
        save_path = new_path + folder
        os.mkdir(save_path)
        images = [i for i in os.listdir(cur_path) if i.endswith(".jpg")]
        # print('1')
        for image_name in images:
            image = imread(f"{cur_path}/{image_name}")
            for i in range(len(cur_bb)):
                x_min, y_min, x_max, y_max = cur_bb[i]
                img = image[y_min:y_max, x_min:x_max]
                save_img_name = f"{save_path}/{image_name.split('.')[0]}_{str(i)}.jpg"
                imsave(save_img_name, img)

    # save to archive
    # os.system('tar -czvf crops.tar.gz /crops')


if __name__ == "__main__":
    path_to_video = "./office/"
    first_room_coords = "left.txt"
    second_room_coords = "right.txt"
    new_path = "./crops/"

    get_data(path_to_video, first_room_coords, second_room_coords, new_path)
