import pandas as pd

data = pd.read_csv('../Data/ClassificationCityNew/city_year_new.csv')
print(data['city'].value_counts())

