import os

import cv2
import numpy as np
import pandas as pd
import ptwt
import pywt
import torch
from numba import prange
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from shutil import rmtree


class VideoDataset(Dataset):
    def __init__(self, path, labels, num_frames, height, width, transforms=None, pickle=False, fft=False, dct=False, wavelet=False):
        self.X = path
        self.y = labels
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.aug = transforms
        self.pickle = pickle
        self.fft = fft
        self.dct = dct
        self.wavelet = wavelet
        assert (self.fft ^ self.dct ^ self.wavelet) or not (self.fft or self.dct or self.wavelet), "Cannot use both fft and dct"
        
    def read_video(self, path):
        
        files = os.listdir(path)
        
        NUM_FRAMES = self.num_frames
        NUM_CHANNELS = 3 
        
        if self.fft:
            NUM_CHANNELS = 6
        elif self.dct:
            NUM_CHANNELS = 3
        elif self.wavelet:
            NUM_CHANNELS = 3
            
        frames = torch.zeros((NUM_FRAMES, NUM_CHANNELS, self.height, self.width))
        
        if self.aug:
            noise, compression, blur, flip, one_of, grey, shift = self.aug.get_transforms()
        
        for idx in prange(len(files)):
            file_path = os.path.join(path, files[idx])
            if self.aug:
                frames[idx] = self.read_file(file_path, noise=noise, compression=compression, blur=blur, flip=flip, one_of=one_of, grey=grey, shift=shift)
            else:
                frames[idx] = self.read_file(file_path)
        
        return frames
    
    def read_file(self, file_path, noise=0, compression=0, blur=0, flip=0, one_of=0, grey=0, shift=0):
        if self.pickle:
            frame = np.load(os.path.join(file_path))
        else:
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.aug:
                frame = self.aug(frame, noise, compression, blur, flip, one_of, grey, shift)
                frame = frame.numpy()
                frame = frame.transpose(1,2,0)
            
        if self.fft or self.dct or self.wavelet:
            if self.fft:
                frame = img_fast_fourier_transform(frame)
            elif self.dct:
                frame = img_discrete_cosine_transform(frame)
            elif self.wavelet:
                frame = img_wavelet_transform(frame)
                
        frame = frame.astype(np.float32)
        frame = frame.transpose(2,0,1)
        frame = torch.from_numpy(frame)
        
        return frame            
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        video = self.read_video(self.X[idx])
        label_t = torch.tensor([self.y[idx]], dtype=torch.float)
        # label for spatial feature
        label_s = torch.tensor([self.y[idx] for _ in range(self.num_frames)], dtype=torch.float)
        return video, label_t, label_s
    
class DataLoaderWrapper(DataLoader):
    def __init__(self, X, y, transforms, num_frames=16, height=224, width=224, batch_size=1, shuffle=False, fft=False, dct=False, wavelet=False):
        dataset = VideoDataset(X, y, height=height, width=width, num_frames=num_frames, transforms=transforms, fft=fft, dct=dct, wavelet=wavelet)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=os.cpu_count())

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img
        
def img_fast_fourier_transform(img):    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    img = np.fft.fftshift(np.fft.fft2(img))
    real = img.real # H X W X C
    imag = img.imag # H X W X C
    real = normalize(real)
    imag = normalize(imag)
    real = np.log(real + 1)
    imag = np.log(imag + 1)
    fourier = np.concatenate((real, imag), axis=2)
    return fourier
    
def img_discrete_cosine_transform(img):
    r = normalize(dct(dct(img[:, :, 0], axis=0), axis=1))
    g = normalize(dct(dct(img[:, :, 1], axis=0), axis=1))
    b = normalize(dct(dct(img[:, :, 2], axis=0), axis=1))
    out = np.stack((r, g, b), axis=2)
    return out

def img_wavelet_transform(img):
    if img.shape[0] != 3:
        img = img.transpose(2,0,1)
        
    if type(img) != torch.Tensor:
        img = torch.from_numpy(img).to(torch.float64)
    

    coeffs = ptwt.wavedec2(img, pywt.Wavelet("haar"), level=2, mode="constant")
    cA, (cAH, cAV, cAD), (cH, cV, cD) = coeffs
    
    cA = normalize(cA).squeeze(dim=1)
    cH = normalize(cH).squeeze(dim=1)
    cV = normalize(cV).squeeze(dim=1)
    cD = normalize(cD).squeeze(dim=1)
    cAH = normalize(cAH).squeeze(dim=1)
    cAV = normalize(cAV).squeeze(dim=1)
    cAD = normalize(cAD).squeeze(dim=1)
    
    top_left = np.zeros((3, cA.shape[1] * 2, cA.shape[2] * 2))
    
    top_left[:, :cA.shape[1], :cA.shape[2]] = cA
    top_left[:, :cA.shape[1], cA.shape[2]:] = cAH
    top_left[:, cA.shape[1]:, :cA.shape[2]] = cAV
    top_left[:, cA.shape[1]:, cA.shape[2]:] = cAD
    
    
    out = np.zeros((3, top_left.shape[1] * 2, top_left.shape[2] * 2))
    
    out[:, :top_left.shape[1], :top_left.shape[2]] = top_left
    out[:, :top_left.shape[1], top_left.shape[2]:] = cH
    out[:, top_left.shape[1]:, :top_left.shape[2]] = cV
    out[:, top_left.shape[1]:, top_left.shape[2]:] = cD
    
    out = out.transpose(1,2,0)
    
    return out

if __name__ == "__main__":
    PATH = './celeb-df/videos_16/data_video.csv'

    df = pd.read_csv(PATH)
    print(df.head())
    X = df['filename'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}") 
    
    train_dataset = DataLoaderWrapper(X_train, y_train, transforms=None, num_frames=16, height=224, width=224, batch_size=2, shuffle=False, fft=False, dct=False, wavelet=False)
    
    for idx, (video, label_t, label_s) in enumerate(train_dataset):
        print(video.shape)
        print(label_t.shape)
        print(label_s.shape)
        
        if idx == 0:
            break
    
    
