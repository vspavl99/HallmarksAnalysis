import pandas as pd
import os


def parse_first_data(path_to_txt=r'Data/Raw data/Выборка_практика_Тлепин/Номер_Дата.txt'):
    data_frame = pd.DataFrame(columns=['image_name', 'city', 'year', 'url'])

    with open(path_to_txt, 'r') as txt:
        for line in txt:
            split_line = line.split("-")
            image_number, other_info = split_line[0], split_line[1]

            other_info_split = other_info.split()

            city, year = other_info_split[0], other_info_split[1]
            url = other_info_split[2] if len(other_info_split) > 2 else None

            row = {
                'image_name': str(image_number).strip() + "_original.jpg",
                'city': city,
                'year': year,
                'url': url
            }

            data_frame = data_frame.append(row, ignore_index=True)

        return data_frame


def parse_second_data(path_data=r'Data/Raw data/Silver hallmarks'):

    data_frame = pd.DataFrame(columns=['folder', 'original_image', 'highlighted_image', 'city', 'year', 'url'])
    original, highlight = None, None

    for folder in os.listdir(path_data):
        for file in os.listdir(os.path.join(path_data, folder)):
            if "original" in file:
                original = file
            if ".txt" in file:

                with open(os.path.join(path_data, folder, file), 'r') as txt:
                    city, year = txt.readline().split()

                    # Pass empty line
                    txt.readline()
                    txt.readline()
                    url = txt.readline().split()[1]

            if "highlighted" in file:
                highlight = file

        new_row = {
            'folder': folder,
            'original_image': original,
            'highlighted_image': highlight,
            'city': city,
            'year': year,
            'url': url
        }

        data_frame = data_frame.append(new_row, ignore_index=True)
    return data_frame


if __name__ == '__main__':
    dataframe1 = parse_first_data()
    print(dataframe1)
    dataframe1.to_csv(r'Data\data_frame_first.csv')

    dataframe2 = parse_second_data()
    print(dataframe2)
    dataframe2.to_csv(r'Data\data_frame_second.csv')


