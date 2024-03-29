import os

import numpy as np
import pandas as pd
from imutils import paths
from tqdm import tqdm

FILE_PATH = 'videos_16/'
SAVE_PATH = 'videos_16/'

image_paths = list(paths.list_images(FILE_PATH))

print(f'[INFO] Found {len(image_paths)} images')

data = pd.DataFrame()

labels = []

for idx, image_path in tqdm(enumerate(image_paths), desc="Creating dataframe"):
    label = image_path.split(os.path.sep)[2]
    data.loc[idx, "image_path"] = image_path
    labels.append(label)

for i in range(len(labels)):    
    data.loc[i, 'label'] = 1 if labels[i] == 'fake' else 0
    
data = data.sample(frac=1).reset_index(drop=True)

print(data.head())

data.to_csv(SAVE_PATH + os.path.sep + 'data.csv', index=False)
