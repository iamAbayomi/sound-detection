# ----------------------------
# Prepare training data from Metadata file
# ----------------------------
# import pandas as pd
# from pathlib import Path

# download_path = Path.cwd()/'UrbanSound8K'

# # Read metadata file
# metadata_file = download_path/'metadata'/'UrbanSound8K.csv'
# df = pd.read_csv(metadata_file)
# df.head()

# # Construct file path by concatenating fold and file name
# df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

# # Take relevant columns
# df = df[['relative_path', 'classID']]
# df.head()


import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
