import pandas as pd
import shutil
import os


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
    # coping_images_into_common_directory()

    data_frame1 = pd.read_csv('Data/data_frame_first.csv')
    data_frame2 = pd.read_csv('Data/data_frame_second.csv')

    data_frame2['image_name'] = data_frame2['folder'] + '_' + data_frame2['original_image']

    data_frame = data_frame1[['image_name', 'city', 'year']].append(
        data_frame2[['image_name', 'city', 'year']], ignore_index=True)

    print(data_frame)

    data_frame.to_csv('Data/annotation.csv')

