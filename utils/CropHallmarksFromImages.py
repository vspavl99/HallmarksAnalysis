import pandas as pd
import numpy as np
import cv2
import os


def extract_hallmark_from_image(original_image: np.ndarray, bbox: tuple) -> np.ndarray:
    """ Cropping the hallmark from image

    :param original_image: Original image with shape (height, width, n_channels)
    :param bbox: tuple containing (left_x, top_y, width, height)
    :return: Cropped hallmark
    """
    left_x, top_y = int(bbox[0]), int(bbox[1])
    right_x, bottom_y = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

    cropped_hallmark = original_image.copy()[top_y:bottom_y, left_x:right_x, :]
    return cropped_hallmark


def created_hallmarks_images(destination_path: str):
    """ Cropping hallmarks from all images in dataset and save them into separate folder

    :return: Don't return anything, Just save images into local disk
    """
    bbox_dataframe = pd.read_csv('../Data/Detection/bounding_boxes.csv')
    city_year_dataframe = pd.read_csv('../Data/ClassificationCity/city_year.csv')

    for _, row in city_year_dataframe.iterrows():

        image = cv2.imread(os.path.join('../Data/Dataset/images/', row['ImageName']))
        assert image is not None, f"Fail! Image not found {row['ImageName']}"

        bounding_boxes = bbox_dataframe[bbox_dataframe['ImageName'] == row['ImageName']]
        for _, bbox_data in bounding_boxes.iterrows():
            print(row['ImageName'])
            hallmark = extract_hallmark_from_image(
                image, (bbox_data['left_x'], bbox_data['top_y'], bbox_data['bbox_width'], bbox_data['bbox_height'])
            )

            cv2.imwrite(os.path.join(destination_path, row['ImageName']), hallmark)


if __name__ == '__main__':

    created_hallmarks_images('../Data/ClassificationCity/images')
