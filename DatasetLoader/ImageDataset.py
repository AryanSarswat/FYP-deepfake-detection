import numpy as np
import pandas as pd
import torch
import albumentations
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image, dtype=torch.float).permute(0, 3, 1, 2)
        labels = torch.tensor(self.y[idx], dtype=torch.long)
        return image, labels
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        dataset = ImageDataset(X, y, transforms=transforms)
        super().__init__(dataset, batch_size, shuffle, num_workers, pin_memory)

if __name__ == "__main__":
    PATH = 'ast_dataset/data_final.csv'

    df = pd.read_csv(PATH)
    X = df['image_path'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    train = DataLoaderWrapper(X_train, y_train, transforms=None)
    test = DataLoaderWrapper(X_test, y_test, transforms=None)
    
    print(f"train: {len(train)}, test: {len(test)}")
    