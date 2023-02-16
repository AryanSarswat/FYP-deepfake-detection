import os

import cv2
import numpy as np
from tqdm import tqdm

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
SRC_PATH_1 = 'faceforensics/manipulated_sequences/DeepFakeDetection/c23'
SRC_PATH_2 = 'faceforensics/manipulated_sequences/Deepfakes/c23'
SRC_PATH_3 = 'faceforensics/manipulated_sequences/Face2Face/c23'
SRC_PATH_4 = 'faceforensics/manipulated_sequences/FaceSwap/c23'
SRC_PATH_5 = 'faceforensics/manipulated_sequences/NeuralTextures/c23'
SRC_PATH_6 = 'faceforensics/manipulated_sequences/FaceShifter/c23'
SRC_PATH = [SRC_PATH_1, SRC_PATH_2, SRC_PATH_3, SRC_PATH_4, SRC_PATH_5, SRC_PATH_6]
DEST_PATH = 'faceforensics/videos_16/'

def preprocess(path, isReal=True, num_frames=32):
    if isReal:
        dest_path = os.path.join(DEST_PATH, 'real')
    else:
        dest_path = os.path.join(DEST_PATH, 'fake')
        
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    
    for dirpath, dirnames, filenames in tqdm(os.walk(path)):
        for filename in filenames:
            if filename.endswith('.mp4'):
                convert_video_to_images(os.path.join(dirpath, filename), os.path.join(dest_path, filename), num_frames=num_frames)

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
    #print("[INFO] Preprocessing real videos...")
    #preprocess(TEST_PATH_2, num_frames=16)
    print("[INFO] Preprocessing fake videos...")
    for i in range(6):
        preprocess(SRC_PATH[i], isReal=False, num_frames=16)
