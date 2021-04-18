import shutil
import pandas as pd
import os


def create_train_val_folder_for_classification(path_to_root: str = '../Data/Classification'):
    for phase in ['train', 'val']:
        dataframe = pd.read_csv(f'{path_to_root}/city_year_{phase}.csv')

        for _, row in dataframe.iterrows():
            image_name = row['ImageName']

            if not os.path.exists(f'{path_to_root}/{phase}'):
                os.mkdir(f'{path_to_root}/{phase}')

            shutil.copy(
                os.path.join(f'{path_to_root}/images', image_name),
                os.path.join(f'{path_to_root}/{phase}', image_name)
            )


def create_train_val_folder_for_detection(path_to_root: str = 'Data/Detection/yolov4/data'):
    for phase in ['train', 'val']:
        dataframe = pd.read_csv(f'../Data/Classification/city_year_{phase}.csv')

        for _, row in dataframe.iterrows():
            image_name = row['ImageName']

            if not os.path.exists(f'{path_to_root}/{phase}'):
                os.mkdir(f'{path_to_root}/{phase}')

            shutil.copy(
                os.path.join(f'{path_to_root}/images', image_name),
                os.path.join(f'{path_to_root}/{phase}', image_name)
            )

            shutil.copy(
                os.path.join(f'{path_to_root}/images', image_name.split('.')[0] + '.txt'),
                os.path.join(f'{path_to_root}/{phase}', image_name.split('.')[0] + '.txt')
            )

# TODO: Combine these two function into one.

