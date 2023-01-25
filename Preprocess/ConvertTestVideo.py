import pandas as pd
import numpy as np
import json

import os

import cv2
import numpy as np
from tqdm import tqdm

#SAMPLE_SUBMISSION_PATH = 'sample_submission.csv'
#df = pd.read_csv(SAMPLE_SUBMISSION_PATH)


FILES_PATH = 'dfdc/train_sample_videos'
DEST_PATH = 'dfdc/videos_16'

df = pd.DataFrame(columns=['filename', 'label'])
METADATA_JSON = 'dfdc/train_sample_videos/metadata.json'

train_files = json.load(open(METADATA_JSON))

for key, value in train_files.items():
    key = key.replace(".mp4",'')
    key = os.path.join(DEST_PATH, key)
    df.loc[len(df)] = (key, 1 if value['label'] == 'FAKE' else 0)
    
print(df.head())

def preprocess(path, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    num_dir = len(os.listdir(path))
    for dirpath, dirnames, filenames in tqdm(os.walk(path), total=num_dir):
        for filename in filenames:
            if filename.endswith('.mp4'):                              
                convert_video_to_images(os.path.join(dirpath, filename), os.path.join(dest_path, filename))

def convert_video_to_images(src_path, dest_path, num_frames=16):
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
    print("[INFO] Preprocessing Test videos...")
    preprocess(FILES_PATH, DEST_PATH)
    df.to_csv("dfdc/videos_16/test_videos.csv", index=False) 
    
