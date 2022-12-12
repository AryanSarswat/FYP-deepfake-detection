import cv2
import numpy as np
from tqdm import tqdm

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'
DEST_PATH = 'videos/'

def preprocess(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in tqdm(filenames):
            if filename.endswith('.mp4'):
                convert_video_to_images(os.path.join(dirpath, filename), os.path.join(DEST_PATH, filename))
                break

def convert_video_to_images(src_path, dest_path, num_frames=64):
    vidcap = cv2.VideoCapture(src_path)
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    idxs = np.linspace(0, total_frames, num=num_frames, endpoint=False, dtype=int)
    
    for idx in idxs:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = vidcap.read()
        
        if not success:
            break
        
        image = cv2.resize(image, (256, 256))
        
        
        path_to_save = os.path.join(dest_path, f'{idx}.jpg')
        cv2.imwrite(path_to_save, image)
        
        
if __name__ == "__main__":
    print("[INFO] Preprocessing real videos...")
    preprocess(REAL_PATH)
    print("[INFO] Preprocessing fake videos...")
    preprocess(FAKE_PATH)
        
