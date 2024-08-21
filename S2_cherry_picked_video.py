from joblib import Parallel, delayed
from pathlib import Path
import os
from pytube import YouTube, cli, Stream
from pytube.innertube import _default_clients
from tqdm import tqdm
from shutil import move
import re

# _default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]
Path(VID_DIR := "data/videos").mkdir(parents=True,exist_ok=True)
[os.remove(f"{VID_DIR}/{file}") for file in os.listdir(VID_DIR) if file.startswith("temp_")]
[os.remove(f"{VID_DIR}/{file}") for file in os.listdir(VID_DIR) if not file.endswith(".mp4")]

def full_width(b) -> str:
    b = list(bytes(b,"utf-16"))
    b[2] -= 32
    b[-1] = 255
    return bytes(b).decode("utf-16")

def sanitize(title) -> YouTube:
    title = re.sub(r'\[[^]]*\]', '', title)
    title = re.sub(r'\【[^]]*\】', '', title)
    for r in "\\/:*?\"<>|#.~'":
        title = title.replace(r, full_width(r)) 
    return title.strip()

class Video():
    def __init__(self, link) -> None:
        self.link = link
        self.title = YouTube(link).title

    def download(self):
        vid_stream = YouTube(self.link).streams.get_highest_resolution()
        vid_stream.download(filename=self.temp_file,skip_existing=False)
        move(self.temp_file, self.mp4_file)

    @property
    def temp_file(self):
        return f"{VID_DIR}/temp_{sanitize(self.title)}.mp4"
    
    @property
    def mp4_file(self):
        return f"{VID_DIR}/{sanitize(self.title)}.mp4"


def download(vid: Video, pb:tqdm) -> None|tuple[str,str]:
    try:
        vid.download()

    except Exception as e:
        with open("logs/Problematic Songs.txt", "a", encoding='utf-8') as file:
            file.write(f"{vid.link}      {type(e)}\n")
            raise e
    finally:
        pb.update(1)

videos = [
    Video("https://youtu.be/jUwDWQPVkA4"),
    Video("https://youtu.be/pqPuewju9Mw"),
]
skip_download = [f"{VID_DIR}/{vid}" for vid in os.listdir(VID_DIR)]
videos = [v for v in videos if v.mp4_file not in skip_download]
# with tqdm(total=len(videos), desc="Downloading videos") as pb:
#     list((download)(v, pb) for v in videos)
#     Parallel(-1, require="sharedmem")(delayed(download)(v, pb) for v in videos)

from pytube.download_helper import (
    download_videos_from_channels,
    download_video,
    download_videos_from_list,
)
download_video("https://www.youtube.com/watch?v=olmYHXHiLGg")