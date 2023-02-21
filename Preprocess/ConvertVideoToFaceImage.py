import gc
import os

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=224, select_larget=True, post_process=False, device=device)


def get_bounding_box(x1, y1, x2, y2):
    y2 += (y2 - y1) / 10
    w = x2 - x1
    h = y2 - y1
    diff_h_w = (h - w) / 2
    x1 -= diff_h_w
    x2 += diff_h_w
    return x1, y1, x2, y2

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
        
        boxes, probs = mtcnn.detect(image)
        xmin, ymin, xmax, ymax = boxes[0]
        xmin, ymin, xmax, ymax = get_bounding_box(xmin, ymin, xmax, ymax)
        crop_frame = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_frame = cv2.resize(crop_frame, (224, 224))
        
        
        path_to_save = os.path.join(dest_path, f'{idx}.jpg')
        cv2.imwrite(path_to_save, crop_frame)
        
if __name__ == "__main__":
    PATH = ''
    DEST_PATH = 'KoDF/videos_32/'
    print("[INFO] Preprocessing real videos...")
    preprocess(PATH, num_frames=32)
    #print("[INFO] Preprocessing fake videos...")
    #preprocess(TEST_PATH_1, num_frames=16)