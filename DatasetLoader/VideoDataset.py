import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    def __init__(self, path, labels, transforms=None):
        self.X = path
        self.y = labels
        self.aug = transforms
        
    def read_video(self, path):
        frames = []
        
        vidcap = cv2.VideoCapture(path)
        
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        idxs = np.linspace(0, total_frames, 16, endpoint=False, dtype=np.int)
        
        for idx in idxs:
            success, image = vidcap.read()
            
            if not success:
                break
            
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.
            image = self.aug(image=image)['image']
            frames.append(image)
        
        vidcap.release()
        return torch.stack(frames)
            
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        video = self.read_video(self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.float)
        labels = torch.unsqueeze(labels, 0)
        return image, labels
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, batch_size=1, shuffle=False):
        dataset = VideoDataset(X, y, transforms=transforms)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    PATH = 'images/data_video.csv'

    df = pd.read_csv(PATH)
    X = df['video_path'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    train = DataLoaderWrapper(X_train, y_train, transforms=None, batch_size=3, shuffle=True)
    test = DataLoaderWrapper(X_test, y_test, transforms=None, batch_size=3, shuffle=True)
    
    for idx, (X, y) in enumerate(train):
        print(X.shape, y.shape)
        break
    
    print(f"train: {len(train)}, test: {len(test)}")
    