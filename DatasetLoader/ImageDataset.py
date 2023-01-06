import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, path, labels, transforms=None):
        self.X = path
        self.y = labels
        self.aug = transforms
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.X[idx])
        if self.aug is not None:
            image = self.aug(image=np.array(image))['image']
        image = torch.tensor(image, dtype=torch.float).permute(2, 1, 0)
        labels = torch.tensor(self.y[idx], dtype=torch.float)
        labels = torch.unsqueeze(labels, 0)
        return image, labels
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, batch_size=1, shuffle=False):
        dataset = ImageDataset(X, y, transforms=transforms)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    PATH = 'images/data.csv'

    df = pd.read_csv(PATH)
    X = df['image_path'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    train = DataLoaderWrapper(X_train, y_train, transforms=None, batch_size=3, shuffle=True)
    test = DataLoaderWrapper(X_test, y_test, transforms=None, batch_size=3, shuffle=True)
    
    for idx, (X, y) in enumerate(train):
        print(X.shape, y.shape)
        break
    
    print(f"train: {len(train)}, test: {len(test)}")
    