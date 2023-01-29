import os

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from scipy.fftpack import dct
from numba import prange 

class VideoDataset(Dataset):
    def __init__(self, path, labels, num_frames, height, width, transforms=None, pickle=False, fft=False, dct=False):
        self.X = path
        self.y = labels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.aug = transforms
        self.pickle = pickle
        self.fft = fft
        self.dct = dct
        assert not (self.fft and self.dct), "Cannot use both fft and dct"
        
    def read_video(self, path):
        
        files = os.listdir(path)
        
        NUM_FRAMES = self.num_frames
        NUM_CHANNELS = 3 
        
        if self.fft:
            NUM_CHANNELS += 6
        elif self.dct:
            NUM_CHANNELS += 3
            
        frames = torch.zeros((NUM_FRAMES, NUM_CHANNELS, self.height, self.width))
        
        for idx in prange(len(files)):
            frames[idx] = self.read_file(files[idx])
        
        return frames
    
    def read_file(self, file_path):
        if self.pickle:
                frame = np.load(os.path.join(path, file_path))
        else:
            frame = cv2.imread(os.path.join(path, file_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        if self.fft:
            to_concat = img_fast_fourier_transform(frame)
        elif self.dct:
            to_concat = img_discrete_cosine_transform(frame)

        # Normalize to 0-1
        frame = frame / 255
            
        if self.fft or self.dct:
            frame = np.concatenate((frame, to_concat), axis=2)
                
        frame = frame.astype(np.float32)
        frame = torch.from_numpy(frame)
        frame = frame.transpose(2,0,1)
        frame = torch.from_numpy(frame)
        
        return frame            
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        video = self.read_video(self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.float)
        labels = torch.unsqueeze(labels, 0)
        return video, labels
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, num_frames=16, height=224, width=224, batch_size=1, shuffle=False, fft=False, dct=False):
        dataset = VideoDataset(X, y, height=height, width=width, num_frames=num_frames, transforms=transforms, fft=fft, dct=dct)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=8)

def normalize_255(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img     
        
def img_fast_fourier_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    img = np.fft.fftshift(np.fft.fft2(img))
    real = img.real # H X W X C
    imag = img.imag # H X W X C
    real = normalize_255(real)
    imag = normalize_255(imag)
    real = np.log(real + 1)
    imag = np.log(imag + 1)
    fourier = np.concatenate((real, imag), axis=2)
    return fourier
    
def img_discrete_cosine_transform(img):
    r = normalize_255(dct(dct(img[:, :, 0], axis=0), axis=1))
    g = normalize_255(dct(dct(img[:, :, 1], axis=0), axis=1))
    b = normalize_255(dct(dct(img[:, :, 2], axis=0), axis=1))
    out = np.stack((r, g, b), axis=2)
    return out

if __name__ == "__main__":
    PATH = 'videos_16/data_video.csv'

    df = pd.read_csv(PATH)
    X = df['video_path'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}") 
    

    for path in tqdm(X):
        num_frames = len(os.listdir(path))
        if (num_frames != 16):
            print(f"[ERROR] Path {path} does not have 16 frames exactly")
    
