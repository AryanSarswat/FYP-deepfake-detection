from tqdm import tqdm
import os 
import pandas as pd

REAL_PATH = 'videos/real/'
FAKE_PATH = 'videos/fake/'

def get_all_videos(path):
    video_paths = []
    dir_path = os.listdir(path)
    
    for dir_name in dir_path:
        print(dir_name)
        video_paths.append(os.path.join(path, dir_name))
    
    return video_paths

REAL_VIDEOS = get_all_videos(REAL_PATH)
FAKE_VIDEOS = get_all_videos(FAKE_PATH)

df = pd.DataFrame(columns=['video_path', 'label'])


print(f"[INFO] Number of Real videos: {len(REAL_VIDEOS)}")
print(f"[INFO] Number of Fake videos: {len(FAKE_VIDEOS)}")


for video in tqdm(REAL_VIDEOS):
    df.loc[len(df)] = [video, 0]
    
for video in tqdm(FAKE_VIDEOS):
    df.loc[len(df)] = [video, 1]


print(df.head())
print("[INFO] Saving as CSV")

df.to_csv('videos/data_video.csv', index=False)
