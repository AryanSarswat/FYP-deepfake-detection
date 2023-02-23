import os
from os import cpu_count
from pathlib import Path

import numpy as np

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2
import torch    
from facenet_pytorch import MTCNN

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=224, select_larget=True, post_process=False, device=device)

def extract_face(img):
    img = img.cvtColor(cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1)).to(device)
    boxes, probs = mtcnn(img)

    if len(boxes) == 0:
        return None
    else:
        xmin, ymin, xmax, ymax = boxes[0]
        x_diff = xmax - xmin
        y_diff = ymax - ymin
        
        x_min = int(xmin - 0.3 * x_diff)
        x_max = int(xmax + 0.3 * x_diff)
        y_min = int(ymin - 0.3 * y_diff)
        y_max = int(ymax + 0.3 * y_diff)
        
        crop_img = img[y_min:y_max, x_min:x_max]
        crop_img = crop_img.cpu().numpy().transpose(1, 2, 0)
        crop_img = cv2.resize(crop_img, (224, 224))
        
    return crop_img
    
def get_video_paths(root_dir):
    video_paths = []
    for ext in ["mp4", "avi", "mov"]:
        video_paths += glob(os.path.join(root_dir, f"*.{ext}"))
    return video_paths

def extract_video(src_path, dest_path, num_frames=32, isReal=False):
    capture = cv2.VideoCapture(src_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)

    filename = src_path.split(os.pathsep)[-1]
    filename = filename.split('.')[0]
    dest_path = os.path.join(dest_path, filename)
    os.makedirs(dest_path, exist_ok=True)
    
    for idx, frame_num in enumerate(idxs):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, image = capture.read()
        
        if not success:
            break
        
 
        image = extract_face(image)

        if image is None:
            break
        
        path_to_save = os.path.join(dest_path, f'{idx}.jpg')
        cv2.imwrite(path_to_save, image)
        
        
if __name__ == "__main__":
    root_dir = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
    video_paths = get_video_paths(root_dir)
    
    dest_dir = Path("KoDF/videos_32")
    isReal = True
    
    if isReal:
        dest_dir = os.path.join(dest_dir, "real")
    else:
        dest_dir = os.path.join(dest_dir, "fake")
    
    with Pool(cpu_count()) as pool:
        with tqdm(total=len(video_paths)) as pbar:
            for v in pool.imap_unordered(partial(extract_video, dest_path=dest_dir, num_frames=32, isReal=isReal), video_paths):
                pbar.update()
