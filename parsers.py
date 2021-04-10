import pandas as pd
import shutil
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


def coping_images_into_common_directory():
    path_to_first_csv = 'Data/data_frame_first.csv'
    path_to_second_csv = 'Data/data_frame_second.csv'
    path_to_final_directory = 'Data/Dataset/images/'
    path_to_first_data = 'Data/Raw data/Выборка_практика_Тлепин/Оригинал'
    path_to_second_data = 'Data/Raw data/Silver hallmarks'

    if not os.path.exists(r'Data/Dataset'):
        os.makedirs(r'Data/Dataset')

    # Process first data_frame
    try:
        shutil.copytree(path_to_first_data, path_to_final_directory)
    except FileExistsError:
        pass
        # Directory already exists, copying not finished

    # Process second data_frame
    data_frame2 = pd.read_csv(path_to_second_csv)

    for i, row in data_frame2.iterrows():
        folder, original_name = row['folder'], row['original_image']
        path_to_image = os.path.join(folder, original_name)

        shutil.copy(
            os.path.join(path_to_second_data, path_to_image),
            os.path.join(path_to_final_directory, f'{folder}_{original_name}')
        )

    return 0


if __name__ == '__main__':
    dataframe1 = parse_first_data()
    print(dataframe1)
    dataframe1.to_csv(r'Data\data_frame_first.csv')

    dataframe2 = parse_second_data()
    print(dataframe2)
    dataframe2.to_csv(r'Data\data_frame_second.csv')

    # coping_images_into_common_directory()

    dataframe1 = pd.read_csv('Data/data_frame_first.csv')
    dataframe2 = pd.read_csv('Data/data_frame_second.csv')

    dataframe2['image_name'] = dataframe2['folder'] + '_' + dataframe2['original_image']

    dataframe = dataframe1[['image_name', 'city', 'year']].append(
        dataframe2[['image_name', 'city', 'year']], ignore_index=True)

    print(dataframe)

    dataframe.to_csv('Data/annotation_raw.csv')
