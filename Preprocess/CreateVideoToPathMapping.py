from tqdm import tqdm
import os 
import pandas as pd

REAL_PATH = '../../../../hdd/data/KoDF/kodf_release/original_videos/'
FAKE_PATH = '../../../../hdd/data/KoDF/kodf_release/synthesized_videos/'

TYPES = os.listdir(FAKE_PATH)

def get_all_videos(path):
    video_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_paths.append(os.path.join(dirpath, filename))
    
    return video_paths

REAL_VIDEOS = get_all_videos(REAL_PATH)
FAKE_VIDEOS = get_all_videos(FAKE_PATH)

df = pd.DataFrame(columns=['video_path', 'label'])

for video in tqdm(REAL_VIDEOS):
    df.loc[len(df)] = [video, 0]
    
for video in tqdm(FAKE_VIDEOS):
    df.loc[len(df)] = [video, 1]
    
df.to_csv('videos/data_video.csv', index=False)