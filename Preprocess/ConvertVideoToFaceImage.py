import os
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2
import torch    
from facenet_pytorch import MTCNN

from tqdm import tqdm




def extract_face(img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=224, select_largest=True, post_process=False, device=device)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    boxes, probs = mtcnn.detect(img)

    try :
        len(boxes)
    except TypeError:
        return None
    if len(boxes) == 0:
        return None
    else:
        xmin, ymin, xmax, ymax = boxes[0]
        x_diff = abs(xmax - xmin)
        y_diff = abs(ymax - ymin)
        
        x_min = 0 if int(xmin - 0.3 * x_diff) < 0 else int(xmin - 0.3 * x_diff)
        x_max = int(xmax + 0.3 * x_diff)
        y_min = 0 if int(ymin - 0.3 * y_diff) < 0 else int(ymin - 0.3 * y_diff)
        y_max = int(ymax + 0.3 * y_diff)
        
        crop_img = img.cpu().numpy()
        crop_img = crop_img[y_min:y_max, x_min:x_max]
        crop_img = cv2.resize(crop_img, (224, 224))
        
    return crop_img
    
def get_video_paths(root_dir):
    video_paths = []
    for path in Path(root_dir).rglob('*.mp4'):
        video_paths.append(str(path))
    return video_paths

def extract_video(src_path, dest_path, num_frames=32, isReal=False):
    try:
        capture = cv2.VideoCapture(src_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
        idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)

        filename = src_path.split(os.path.sep)[-1]
        filename = filename.split('.')[0]
        dest_path = os.path.join(dest_path, filename)
        os.makedirs(dest_path, exist_ok=True)
        
        idx = 0

        for frame_num in range(int(total_frames)):
            ret = capture.grab()
            
            if not ret:
                print('Error grabbing frame {} from {}'.format(frame_num, src_path))
                break
            
            if frame_num >= idxs[idx]:
                success, image = capture.retrieve()
                
                if not ret or image is None:
                    print('Error retrieving frame {} from {}'.format(frame_num, src_path))
                    break
                else:
                    image_extract = extract_face(image)

                    if image_extract is None:
                        print('Error extracting face from frame {} from {}'.format(frame_num, src_path))
                        image_extract = image
                        image_extract = cv2.resize(image_extract, (224, 224))
                
                    path_to_save = os.path.join(dest_path, f'{idx}.jpg')
                    image = cv2.cvtColor(image_extract, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path_to_save, image)
                
                idx += 1
                if idx >= num_frames:
                    break

    except Exception as e:
        print(src_path)
        print(e)
        
        
if __name__ == "__main__":
    
    real_paths = get_video_paths("../../../../hdd/data/Celeb-DF/Celeb-real")
    real_paths.extend(get_video_paths("../../../../hdd/data/Celeb-DF/Youtube-real"))
    fake_paths = get_video_paths("../../../../hdd/data/Celeb-DF/Celeb-synthesis")
    
    dest_dir = "celeb-df/videos_16"
    isReal = True
    
    if isReal:
        dest_dir = os.path.join(dest_dir, "real")
    else:
        dest_dir = os.path.join(dest_dir, "fake")
    
    with Pool(cpu_count() - 4) as p:
       with tqdm(total=len(real_paths)) as pbar:
           for v in p.imap_unordered(partial(extract_video, dest_path=dest_dir, num_frames=16, isReal=isReal), real_paths):
               pbar.update()
    
    # for path in tqdm(real_paths):
    #     extract_video(path, dest_dir, num_frames=16, isReal=isReal)
                
    # FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
    
    isReal = False
    # video_paths = get_video_paths(FAKE_PATH)
    dest_dir = "celeb-df/videos_16"

    if isReal:
        dest_dir = os.path.join(dest_dir, "real")
    else:
        dest_dir = os.path.join(dest_dir, "fake")
    
    print("Processing Fake Videos")
    with Pool(cpu_count() - 4) as p:
        with tqdm(total=len(fake_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, dest_path=dest_dir, num_frames=16, isReal=isReal), fake_paths):
                pbar.update()
    
