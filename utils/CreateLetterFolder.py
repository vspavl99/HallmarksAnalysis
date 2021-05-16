import os
import cv2
import matplotlib.pyplot as plt


PATH_TO_IMAGES = r'../Data/Detection/yolov4/data/images'


def get_letter_coordinates(image_name: str) -> list:
    txt_name = image_name.split('.')[0] + '.txt'
    raw_coordinates = []
    line_with_letter_coordinates = ''
    with open(os.path.join(PATH_TO_IMAGES, txt_name), 'r') as txt_file:
        for line in txt_file.readlines():
            if line.startswith('1'):
                line_with_letter_coordinates = line
                break

    if line_with_letter_coordinates:
        raw_coordinates = list(map(float, line_with_letter_coordinates.split()[1:]))

    return raw_coordinates


for image_name in os.listdir(PATH_TO_IMAGES):
    if image_name.startswith('.'):
        continue
    if '.txt' in image_name:
        continue

    coordinates = get_letter_coordinates(image_name)
    if len(coordinates) != 4:
        continue

    x_scaled, y_scaled, w_scaled, h_scaled = coordinates

    image = cv2.imread(os.path.join(PATH_TO_IMAGES, image_name))

    image_width, image_height = image.shape[1], image.shape[0]

    x_from = int(x_scaled * image_width - w_scaled * image_width / 2)
    x_to = int(x_scaled * image_width + w_scaled * image_width / 2)
    y_from = int(y_scaled * image_height - w_scaled * image_height / 2)
    y_to = int(y_scaled * image_height + w_scaled * image_height / 2)

    letter = image[y_from:y_to, x_from:x_to, :]

    letter = cv2.resize(letter, (70, 70))

    # plt.imshow(letter)

    cv2.imwrite(os.path.join(PATH_TO_IMAGES.replace('images', 'letters'), image_name), letter)
    print(os.path.join(PATH_TO_IMAGES.replace('images', 'letters'), image_name))
    # plt.show()

