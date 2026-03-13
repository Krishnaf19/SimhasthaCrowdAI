import pandas as pd
df = pd.read_csv('simhastha_master_index.csv')
train = df[df['split_assignment'] == 'Train']
print(train[['image_name','head_count']].to_string())
print()
print(train['head_count'].describe())
