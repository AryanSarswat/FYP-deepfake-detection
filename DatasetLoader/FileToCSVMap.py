import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from imutils import paths

FILE_PATH = 'ast_dataset/data_final.csv'
SAVE_PATH = 'ast_dataset/data_final'


image_paths = list(paths.list_images(FILE_PATH))

data = pd.DataFrame()

labels = []

for idx, image_path in tqdm(enumerate(image_paths), desc="Creating dataframe"):
    label = image_path.split(os.path.sep)[0]
    data.loc[idx, "image_path"] = image_path
    labels.append(label)
    
labels = np.array(labels)
lb = LabelBinarizer()
lb.fit(labels)
labels = lb.transform(labels)

print(f"Number of classes found : {lb.classes_}")

for i in range(len(labels)):
    data.loc[i, 'label'] = int(np.argmax(labels[i]))
    
data = data.sample(frac=1).reset_index(drop=True)

print(data.head())

data.to_csv(SAVE_PATH + os.path.sep + '.csv', index=False)


# pickle the binarized labels
print('Saving the binarized labels as pickled file')
joblib.dump(lb, SAVE_PATH + os.path.sep + 'lb.pkl')