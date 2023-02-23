import os

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Optimisations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#! Hyper parameters
# Debugging
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def get_bounding_box(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    x1 -= w // 3
    x2 += w // 3
    y1 -= h // 3
    y2 += h // 3 
    return x1, y1, x2, y2

def convert_video_to_images(src_path, dest_path, num_frames=32, device=None):
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
        
        temp = torch.from_numpy(image)
        temp = temp.to(device)
        boxes, probs = mtcnn.detect(temp)
        xmin, ymin, xmax, ymax = boxes[0]
        xmin, ymin, xmax, ymax = get_bounding_box(xmin, ymin, xmax, ymax)
        crop_frame = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_frame = cv2.resize(crop_frame, (224, 224))
        
        
        path_to_save = os.path.join(dest_path, f'{idx}.jpg')
        cv2.imwrite(path_to_save, crop_frame)
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=224, select_largest=True, post_process=False, device=device)
    PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
    PATH_FS = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/fo'
    PATH_FO = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/fsgan'
    DEST_PATH = 'KoDF/videos_32/'
    isReal = False

    files = []


    for dirpath, dirnames, filenames in os.walk(PATH_FO):
        for filename in filenames:
            if filename.endswith('.mp4'):
                files.append(os.path.join(dirpath, filename))

    dest_path = os.path.join(DEST_PATH, 'real') if isReal else os.path.join(DEST_PATH, 'fake')

    os.makedirs(dest_path, exist_ok=True)


    for src_path in tqdm(files):
        dst_file_path = os.path.join(dest_path, src_path.split('/')[-1])
        try:
            convert_video_to_images(src_path, dst_file_path, num_frames=32)
        except Exception as e:
            print(f"[ERROR] Can't process file {src_path}")
            print(e)
            dst_file_path = dst_file_path.replace('.mp4', '')
            os.rmdir(dst_file_path)
            
