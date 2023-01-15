import cv2
import os
import numpy as np
from tqdm import tqdm
import typing
from scipy.fftpack import dct

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
DEST_PATH = 'videos16_pickled/'

def preprocess(path: str, is_real: bool = True, num_frames: int = 32, fft: bool = False, dct: bool = False):
    if is_real:
        dest_path = os.path.join(DEST_PATH, 'real')
    else:
        dest_path = os.path.join(DEST_PATH, 'fake')
        
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    num_dir = len(os.listdir(path))
    for dirpath, dirnames, filenames in tqdm(os.walk(path), total=num_dir):
        for filename in filenames:
            if filename.endswith('.mp4'):                              
                convert_video_to_images(os.path.join(dirpath, filename), os.path.join(dest_path, filename), num_frames=num_frames, fft=fft, dct=dct)

def convert_video_to_images(src_path: str, dest_path: str, num_frames: int = 32, fft: bool = False, dct: bool = False):
    # Load Videos
    vidcap = cv2.VideoCapture(src_path)
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frame number to extract
    idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)
    
    dest_path = dest_path.replace('.mp4', '.npz')
    
    frames = []

    for idx, frame_num in enumerate(idxs):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        
        if not success:
            print(f"[INFO] Error reading frame from {src_path}")
            break
        
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if fft:
            fourier_img = img_fast_fourier_transform(image)
        if dct:
            dct_img = img_discrete_cosine_transform(image)
        if fft:
            image = np.concatenate((image, fourier_img), axis=2)
        if dct: 
            image = np.concatenate((image, dct_img), axis=2)

        frames.append(image) 
        
    to_save = np.stack(frames, axis=2)
    np.savez_compressed(dest_path, to_save)

def img_fast_fourier_transform(img):
    img = np.fft.fftshift(np.fft.fft2(img))
    real = img.real # H X W X C
    imag = img.imag # H X W X C
    real = normalize_255(real)
    imag = normalize_255(imag)
    fourier = np.concatenate((real, imag), axis=2)
    return fourier
    
def img_discrete_cosine_transform(img):
    r = normalize_255(dct(dct(img[:, :, 0], axis=0), axis=1))
    g = normalize_255(dct(dct(img[:, :, 1], axis=0), axis=1))
    b = normalize_255(dct(dct(img[:, :, 2], axis=0), axis=1))
    dct = np.stack((r, g, b), axis=2)
    return dct

def normalize_255(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img
        
if __name__ == "__main__":
    print("[INFO] Preprocessing real videos...")
    preprocess(REAL_PATH, num_frames=16, fft=True)
    print("[INFO] Preprocessing fake videos...")
    preprocess(FAKE_PATH, num_frames=16, isReal=False, fft=True)
