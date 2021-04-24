import os
import cv2
import pandas as pd
from utils.visualisation import draw_boxes


def parse_bbox_line(bbox_line: str) -> tuple:
    """ Extract from string bounding box information

    :param bbox_line: String like " 99%   (left_x:  369   top_y:  594   width:  193   height:  110)"
    :return: Extracted percent of confidence, Coords of left top point, and width and height of Bounding Box
    """

    percent = bbox_line[:bbox_line.find('(')].strip()
    left_x = bbox_line.find(':')
    top_y = bbox_line.find(':', left_x + 1)
    width = bbox_line.find(':', top_y + 1)
    height = bbox_line.find(':', width + 1)

    left_x = bbox_line[left_x + 1:left_x + 9]
    top_y = bbox_line[top_y + 1:top_y + 9]
    width = bbox_line[width + 1:width + 9]
    height = bbox_line[height + 1:height + 6]

    return percent, int(left_x), int(top_y), int(width), int(height)


def txt_to_csv(path_to_txt: str) -> pd.DataFrame:
    """ Parsing YoloV4 result.txt file into DataFrame.csv file with columns
        ('ImagePath', 'confidence', 'left_x', 'top_y', 'width', 'height')

    :param path_to_txt: Path from current content root to .txt file with YoloV4 predictions
    :return: DataFrame
    """

    data = pd.DataFrame(columns=['ImagePath', 'confidence', 'left_x', 'top_y', 'width', 'height'])
    boxes = []

    with open(path_to_txt, 'r') as txt_file:
        for line in txt_file:

            if 'Predicted' in line:
                for percent, left_x, top_y, width, height in boxes:
                    data = data.append({
                            'ImagePath': path_to_image,
                            'confidence': percent,
                            'left_x': left_x,
                            'top_y': top_y,
                            'width': width,
                            'height': height
                        }, ignore_index=True)

                boxes = []
                path_to_image = line.split(':')[0]
            elif 'hallmark' in line:
                sub_line = line[9:-2]
                boxes.append((parse_bbox_line(sub_line)))

    return data


if __name__ == '__main__':
    """ This file parse YoloV4 result file into csv format. Then visualize bounding box on image"""

    result_csv = txt_to_csv('result.txt')

    result_csv.to_csv('Data/Detection/bounding_boxes_predicted_yolo.csv')

    for path in result_csv['ImagePath'].unique():
        sub_data = result_csv[result_csv['ImagePath'] == path]

        image = cv2.imread(os.path.join('../Data/Detection/yolov4/data/images', path.split('/')[-1]))
        assert image is not None, "Error! Image not read!"

        for num, row in sub_data.iterrows():
            params = {
                'image': image,
                'percent': row['confidence'],
                'left_x': row['left_x'],
                'top_y': row['top_y'],
                'width': row['width'],
                'height': row['height']
            }
            image = draw_boxes(**params)

        cv2.imwrite(os.path.join('../Data/Detection/yolov4/data/val_result', path.split('/')[-1]), image)

