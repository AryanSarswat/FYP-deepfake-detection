import os

import pandas as pd
from tqdm import tqdm

REAL_PATH = 'celeb-df/videos_16/real/'
FAKE_PATH = 'celeb-df/videos_16/fake/'

def get_all_videos(path):
    video_paths = []
    dir_path = os.listdir(path)
    
    for dir_name in dir_path:
        print(dir_name)
        video_paths.append(os.path.join(path, dir_name))
    
    return video_paths

REAL_VIDEOS = get_all_videos(REAL_PATH)
FAKE_VIDEOS = get_all_videos(FAKE_PATH)

df = pd.DataFrame(columns=['filename', 'label'])


print(f"[INFO] Number of Real videos: {len(REAL_VIDEOS)}")
print(f"[INFO] Number of Fake videos: {len(FAKE_VIDEOS)}")


for video in tqdm(REAL_VIDEOS):
    df.loc[len(df)] = [video, 0]
    
for video in tqdm(FAKE_VIDEOS):
    df.loc[len(df)] = [video, 1]


print(df.head())
print("[INFO] Saving as CSV")

df.to_csv('celeb-df/videos_16/data_video.csv', index=False)
