import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
            frames.append(frame)
        
        to_tensor = transforms.ToTensor()
        frames = torch.stack([to_tensor(frame) for frame in frames])
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
    
    train = DataLoaderWrapper(X_train, y_train, transforms=None, batch_size=2, shuffle=True)
    test = DataLoaderWrapper(X_test, y_test, transforms=None, batch_size=2, shuffle=True)
    
    for idx, (X, y) in enumerate(train):
        print(f"Shape of input {X.shape}, Shape of label {y.shape}")
        print(f"Type of input {X.dtype}, Type of label {y.dtype}")
        print(f"Min of input {X.min()}, Max of input {X.max()}")
        break
    
    print(f"train: {len(train)}, test: {len(test)}")
    