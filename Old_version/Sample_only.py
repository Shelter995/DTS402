import pandas as pd

# 1. Read the original large dataset.
df = pd.read_csv('D:/Projects/DTS402/dataset/shuffleFRDDS.csv')

print(f"Size of the original dataset: {df.shape}")

# 2. Extract 3,000 items.
df_subset = df.sample(n=1000)

# 3. Reset the index
df_subset = df_subset.reset_index(drop=True)

print(f"Size of the new dataset: {df_subset.shape}")

# 4. Save as a new file
# index=False: The representative does not save these index numbers such as 0, 1, 2... into the CSV file.
df_subset.to_csv('D:/Projects/DTS402/dataset/FRDDS_dataset_1k.csv', index=False)

print("Saved successfully! The file name is FRDDS_dataset_1k.csv")