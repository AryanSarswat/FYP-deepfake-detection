import cv2
import os
import numpy as np
from tqdm import tqdm
import typing
import matplotlib.pyplot as plt 

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
DEST_PATH = 'videos_pickled/'

def preprocess(path: str, is_real: bool = True):
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
                convert_video_to_images(os.path.join(dirpath, filename), os.path.join(dest_path, filename))

def convert_video_to_images(src_path: str, dest_path: str, num_frames: int = 32, fft: bool = False, dct: bool = False):
    # Load Videos
    vidcap = cv2.VideoCapture(src_path)
    
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get the frame number to extract
    idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)
    
    dest_path = dest_path.replace('.mp4', '')
    os.makedirs(dest_path, exist_ok=True)
    
    for idx, frame_num in enumerate(idxs):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        
        if not success:
            print(f"[INFO] Error reading frame from {src_path}")
            break
        
        image = cv2.resize(image, (256, 256))
        
        if fft:
            image = img_fast_fourier_transform(image)
            
        if dct:
            image = img_discrete_cosine_transform(image)

        path_to_save = os.path.join(dest_path, f'{idx}.npy')
        np.save(path_to_save, image)
        
        
def img_fast_fourier_transform(img):
    img = np.fft.fftshift(np.fft.fft2(img))
    img = abs(img)
    img = np.log(img + 1e-10) # to avoid log(0)
    img = (img - img.min()) / (img.max() - img.min()) # normalize
    return img
    
def img_discrete_cosine_transform(img):
    #img = cv2.imread(img_path)
    return np.fft.dct(img)
        
if __name__ == "__main__":
    #print("[INFO] Preprocessing real videos...")
    #preprocess(REAL_PATH)
    #print("[INFO] Preprocessing fake videos...")
    #preprocess(FAKE_PATH, isReal=False)
    
    img = cv2.imread("C:\\Users\\Razer\\Pictures\\Saved Pictures\\minimalist-rock-climbing-a1910.jpg")
    img = cv2.resize(img, (256, 256))
    img = img_fast_fourier_transform(img)
    plt.imshow(img)
    plt.show()
    
    
