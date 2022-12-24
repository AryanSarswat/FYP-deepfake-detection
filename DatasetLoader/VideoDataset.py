import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from tqdm import tqdm
import albumentations

class VideoDataset(Dataset):
    def __init__(self, path, labels, num_frames, transforms=None):
        self.X = path
        self.y = labels
        self.num_frames = num_frames
        self.aug = transforms
        
    def read_video(self, path):
        frames = []
        
        files = os.listdir(path)
        for file in files:
            frame = cv2.imread(os.path.join(path, file))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.transpose(2,0,1)
            frame = frame / 255
            frame = torch.tensor(frame, dtype=torch.float)
            frames.append(frame)
        
        frames = torch.stack(frames)
        return frames
            
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        video = self.read_video(self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.float)
        labels = torch.unsqueeze(labels, 0)
        return video, labels
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, stride=128, batch_size=1, shuffle=False):
        dataset = VideoDataset(X, y, stride, transforms=transforms)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    PATH = 'videos/data_video.csv'

    df = pd.read_csv(PATH)
    X = df['video_path'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
   
    transforms = albumentations.Compose([
        albumentations.Resize(256,256),
        albumentations.Normalize(),
    ])

    train = DataLoaderWrapper(X_train, y_train, transforms=transforms, batch_size=128, shuffle=True)
    test = DataLoaderWrapper(X_test, y_test, transforms=transforms, batch_size=128, shuffle=True)
    
    for idx, (X_sample, y_sample) in tqdm(enumerate(train), total=len(train)):
        print(f"[INFO] Shape of training data : {X_sample.shape}")
        print(f"[INFO] Min of training data : {torch.min(X_sample)}")
        print(f"[INFO] Max of training data : {torch.max(X_sample)}")
        print(f"[INFO] Training data type : {X_sample.dtype}")
        print(f"[INFO] Training label type : {y_sample.dtype}")
        break
    
    print(df.head())

    for path in tqdm(X):
        num_frames = len(os.listdir(path))
        if (num_frames != 32):
            print(f"[ERROR] Path {path} does not have 32 frames exactly")
    print(f"train: {len(train)}, test: {len(test)}")
    