import pandas as pd
import cv2
import matplotlib.pyplot as plt


def parse_bbox_line(bbox_line):

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


def txt_to_csv(path_to_txt):
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


def draw_text_with_background(image, text, position):
    font_scale = 2.5
    font = cv2.FONT_HERSHEY_PLAIN

    rectangle_background_color = (255, 0, 255)

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


def draw_boxes(path_to_image, percent, left_x, top_y, width, height):
    print(path_to_image, left_x, top_y, width, height)
    image = cv2.imread(path_to_image)

    assert image is not None, "Fail! Image don't read"

    image = draw_text_with_background(image, f'Hallmark: {percent}', (left_x, top_y))
    image = cv2.rectangle(image, (left_x, top_y), (left_x + width, top_y + height), (255, 0, 255))
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    result_csv = txt_to_csv('utils/result.txt')

    for path in result_csv['ImagePath'].unique():
        sub_data = result_csv[result_csv['ImagePath'] == path]
        for num, row in sub_data.iterrows():
            draw_boxes(*row)


