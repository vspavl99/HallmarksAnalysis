import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
    # Split data into a train and test

    data_frame = pd.read_csv(
        '../Data/Classification/CleanedData/city_year.csv', usecols=['ImageName', 'city', 'year']
    )

    data_frame_train, data_frame_val = train_test_split(
        data_frame, test_size=0.25, random_state=1,
        stratify=data_frame['city']
    )

    data_frame_train.to_csv('../Data/Classification/CleanedData/city_year_train.csv', index=False)
    data_frame_val.to_csv('../Data/Classification/CleanedData/city_year_val.csv', index=False)

    create_train_val_folder_for_classification('../Data/Classification/CleanedData')