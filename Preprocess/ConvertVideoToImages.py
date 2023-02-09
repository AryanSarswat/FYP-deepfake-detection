import os

import cv2
import numpy as np
from tqdm import tqdm

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
TEST_PATH_1 = 'celeb-df/Celeb-real/'
TEST_PATH_2 = 'celeb-df/YouTube-real'
TEST_PATH_3 = 'celeb-df/Celeb-synthesis'
DEST_PATH = 'celeb-df/videos_16/'

def preprocess(path, isReal=True, num_frames=32):
    if isReal:
        dest_path = os.path.join(DEST_PATH, 'real')
    else:
        dest_path = os.path.join(DEST_PATH, 'fake')
        
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    
    num_dir = len(os.listdir(path))
    for filename in tqdm(os.listdir(path), total=num_dir):
        convert_video_to_images(os.path.join(path, filename), os.path.join(dest_path, filename), num_frames=num_frames)

def convert_video_to_images(src_path, dest_path, num_frames=32):
    vidcap = cv2.VideoCapture(src_path)
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)
    
    dest_path = dest_path.replace('.mp4', '')
    os.makedirs(dest_path, exist_ok=True)
    
    for idx, frame_num in enumerate(idxs):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = vidcap.read()
        
        if not success:
            break
        
        image = cv2.resize(image, (224, 224))

        path_to_save = os.path.join(dest_path, f'{idx}.jpg')
        cv2.imwrite(path_to_save, image)
        
        
if __name__ == "__main__":
    print("[INFO] Preprocessing real videos...")
    preprocess(TEST_PATH_1, num_frames=16)
    preprocess(TEST_PATH_2, num_frames=16)
    #print("[INFO] Preprocessing fake videos...")
    #preprocess(TEST_PATH_3, isReal=False, num_frames=16)
