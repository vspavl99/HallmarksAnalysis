import pandas as pd

data = pd.read_csv('../Data/ClassificationYear/data.csv')

# Clear digit from dataset
data = data[data['target'] > 9]
data['target'] = data['target'] - 10


# combine capital and lowercase letters
data['target'] = data['target'].map(lambda x: x - 26 if x >= 26 else x)

# Save the result
data.to_csv('newData.csv')


