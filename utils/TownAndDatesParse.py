import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

classes = ['sheffield', 'sterling', 'date', 'maker', 'george_3', 'london', 'chester',
           'birmingham', 'edinbourgh_sterling', 'edinburgh', 'dublin_sterling', 'dublin', 'victoria', 'glasgow',
           'glasgow_sterling', 'newcastle', 'george_4', 'exeter', 'new_sterling', 'willian_4', 'brittania_958',
           'sterlingr', 'maker]']

correct = [0, 5, 6, 7, 9, 11, 13, 15, 17]

num_classes = list(zip(classes, list(range(len(classes)))))

data_frame = pd.DataFrame(columns=['image_name', 'city'])
path = r'C:\Users\vpavl\Downloads\Telegram Desktop\towns_and_dates\annotations'

for name in os.listdir(path):
    if name == '.DS_Store' or name == 'classes.txt':
        continue

    x_left = 1000
    x_right = 0
    y_left = 1000
    y_right = 0

    row = {}
    with open(os.path.join(path, name), 'r') as txt:

        image = cv2.imread(os.path.join(path, '../..', name.replace('.txt', '.jpg')))

        for num, line in enumerate(txt):
            try:
                x, y, w, h = line[2:].split()
                x, y, w, h = float(x) * 500, float(y) * 500, float(w) * 500, float(h) * 500
                x, y, w, h = int(x), int(y), int(w), int(h)

            except ValueError:
                print(line)

            # print(x, y, w, h)
            # image = cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (200, 0, 200), 2)
            x_left = min(int(float(x - w // 2)), x_left)
            y_left = min(int(float(y - h // 2)), y_left)

            x_right = max(x + w // 2, x_right)
            y_right = max(y + h // 2, y_right)

            if int(line.split()[0]) in correct:
                city = line.split()[0]

        try:
            row = {
                'ImageName': name.replace('.txt', '.jpg'),
                'city': classes[int(city)]
            }
        except NameError:
            print(name)

        # image = cv2.rectangle(image, (x_left, y_left), (x_right, y_right), (0, 200, 200), 2)
        image = image[y_left:y_right, x_left:x_right]

        if image.shape[0] > image.shape[1]:
            image = np.rot90(image)

        cv2.imwrite(os.path.join(path, '../hallmark/', name.replace('.txt', '.jpg')), image)
        # plt.imshow(image)
        # plt.show()
        # print(x_left, y_left, x_right, y_right)

    # row = {
    #     'image_name': str(image_number).strip() + "_original.jpg",
    #     'city': city,
    #     'year': year,
    #     'url': url
    # }

    data_frame = data_frame.append(row, ignore_index=True)


data_frame.to_csv('test.csv')

res = pd.concat((pd.read_csv('../Data/ClassificationCity/city_year.csv').drop(columns='year'), data_frame), axis = 0, in)
res.to_csv('city_year_new.csv')