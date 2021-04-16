import pandas as pd


def parse_bbox_line(bbox_line):

    left_x = bbox_line.find(':')
    top_y = bbox_line.find(':', left_x + 1)
    width = bbox_line.find(':', top_y + 1)
    height = bbox_line.find(':', width + 1)

    left_x = bbox_line[left_x + 1:left_x + 9]
    top_y = bbox_line[top_y + 1:top_y + 9]
    width = bbox_line[width + 1:width + 9]
    height = bbox_line[height + 1:height + 6]

    return int(left_x), int(top_y), int(width), int(height)


def txt_to_csv(path_to_txt):
    data = pd.DataFrame(columns=['ImagePath', 'left_x', 'top_y', 'width', 'height'])
    boxes = []

    with open(path_to_txt, 'r') as txt_file:
        for line in txt_file:

            if 'Predicted' in line:
                for left_x, top_y, width, height in boxes:
                    data = data.append({
                            'ImagePath': path_to_image,
                            'left_x': left_x,
                            'top_y': top_y,
                            'width': width,
                            'height': height
                        }, ignore_index=True)

                boxes = []
                path_to_image = line.split(':')[0]
            elif 'hallmark' in line:
                sub_line = line[15:-2]
                boxes.append((parse_bbox_line(sub_line)))

    return data

# test
print(txt_to_csv('utils/result.txt'))
