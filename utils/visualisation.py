import cv2
import numpy as np


def draw_text_with_background(image: np.ndarray, text: str,
                              position: tuple, rectangle_background_color: tuple = (255, 0, 255)) -> np.ndarray:
    """ Write text with background on image

    :param rectangle_background_color: color of background
    :param image: Image represented in numpy-array format
    :param text: String that should be written on image
    :param position: Coordinate (x, y) of position of the beginning of text
    :return: Image with write text
    """

    font_scale = 2.5
    font = cv2.FONT_HERSHEY_PLAIN

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]

    # set the text start position
    text_offset_x = position[0]
    text_offset_y = position[1]

    # make the coords of the box with a small padding of two pixels
    background_coords = (
        (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2)
    )

    cv2.rectangle(image, background_coords[0], background_coords[1], rectangle_background_color, cv2.FILLED)
    cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)

    return image


def draw_boxes(image: np.ndarray, percent: str, class_name: str, left_x: int, top_y: int,
               width: int, height: int, with_percent: bool = True) -> np.ndarray:
    """ Draw bounding boxes with class name on image

    :param image: Original image
    :param percent: Percent of confidence
    :param class_name: Name of class
    :param left_x: x-coordinate of top left corner of bounding box
    :param top_y: y-coordinate of top left corner of bounding box
    :param width: width of bounding box in pixels
    :param height: height of bounding box in pixels
    :param with_percent: boolean param
    :return: Image with bounding box and class name of object
    """

    text = f'{class_name}: {percent}' if with_percent else f'{class_name}'

    color = (255, 0, 255) if class_name == 'hallmark' else (255, 255, 0)
    image = draw_text_with_background(image, text, (left_x, top_y), color)
    image = cv2.rectangle(image, (left_x, top_y), (left_x + width, top_y + height), color, 2)

    return image
