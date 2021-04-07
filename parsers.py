import pandas as pd


def parse_first_data(path_to_txt=r'C:\Файлы и программы\Учеба\ВКР\Данные\Выборка_практика_Тлепин\Номер_Дата.txt'):
    data_frame = pd.DataFrame(columns=['image_name', 'city', 'year', 'url'])

    with open(path_to_txt, 'r') as txt:
        for line in txt:
            split_line = line.split("-")
            image_number, other_info = split_line[0], split_line[1]

            other_info_split = other_info.split()

            city, year = other_info_split[0], other_info_split[1]
            url = other_info_split[2] if len(other_info_split) > 2 else None

            row = {
                'image_name': str(image_number) + "_original.jpg",
                'city': city,
                'year': year,
                'url': url
            }

            data_frame = data_frame.append(row, ignore_index=True)

        return data_frame


if __name__ == '__main__':
    dataframe = parse_first_data()
    print(dataframe)
    dataframe.to_csv(r'Data\data_frame_first.csv')
