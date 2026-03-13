import pandas as pd

df = pd.read_csv('simhastha_master_index.csv')

# Move the 1050-person image to Inference instead of Train
outlier = 'Simhastha_Kumbh_Mela_at_Nashik_in_Maharashtra_state.jpg'
mask = df['image_name'] == outlier
df.loc[mask, 'split_assignment'] = 'Inference'

df.to_csv('simhastha_master_index.csv', index=False)

# Confirm
train = df[df['split_assignment'] == 'Train']
print(f"Training images now: {len(train)}")
print(f"Max count in train : {train['head_count'].max()}")
print(f"Mean count in train: {train['head_count'].mean():.1f}")
