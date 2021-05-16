import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../Data/ClassificationYear/data.csv')

# Clear digit from dataset
data = data[data['target'] > 9]
data['target'] = data['target'] - 10


# combine capital and lowercase letters
data['target'] = data['target'].map(lambda x: x - 26 if x >= 26 else x)


# split into train and test
data_frame_train, data_frame_val = train_test_split(
    data, test_size=0.25, random_state=1,
    stratify=data['target']
)

# Save the result
data_frame_train.to_csv('newData_train.csv')
data_frame_val.to_csv('newData_val.csv')
data.to_csv('newData.csv')
