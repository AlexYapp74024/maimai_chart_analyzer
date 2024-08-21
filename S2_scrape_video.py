from joblib import Parallel, delayed
from pathlib import Path
import sqlite3
import os
from youtubesearchpython import VideosSearch
from pytube import YouTube, cli, Stream
from pytube.innertube import _default_clients
from tqdm import tqdm
from shutil import move

_default_clients["ANDROID_MUSIC"] = _default_clients["ANDROID_CREATOR"]
CHART_DB_NAME = "charts.db"
VIDEO_DB_NAME = "videos.db"
if not os.path.exists(VIDEO_DB_NAME):
    db = sqlite3.connect(VIDEO_DB_NAME, isolation_level=None)
    db.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY,
        song_name VARCHAR(50),
        publish_date DATE,
        vid_name VARCHAR(100),
        link VARCHAR(100)
    )''')
    db.close()

with open("logs/No vids found.txt", "w", encoding='utf-8'): pass
with open("logs/Problematic Songs.txt", "w", encoding='utf-8'):pass

def get_charts():
    """We only go for lv 13 and greater"""

    db = sqlite3.connect(CHART_DB_NAME, isolation_level=None)

    statement = ''' SELECT DISTINCT s.name, c.difficulty, c.level FROM charts c INNER JOIN songs s
                    ON c.song_id = s.id
                    WHERE (c.difficulty = "MASTER" or c.difficulty = "Re:MASTER") 
                        AND (c.level LIKE "%15%" OR c.level LIKE "%14%" OR c.level LIKE "%13%")
                    ORDER BY c.level DESC
    '''
    return [s for s in db.execute(statement)]

def search_video(name:str, diff:str="Mas") -> list[tuple[str]]:
    result = VideosSearch(f'maimai {name.replace("-","")} {diff.title()}', limit = 20).result()['result']

    def contains(string:str, ls:list[str]):
        return any([l.lower() in string.lower() for l in ls])

    exclude_keyword = ["直撮り", "Music Video", "Expert"]
    include_keyword = ["外部出力", "譜面確認用"]
    result = [r for r in result if not contains(r["title"], exclude_keyword)]
    result = [r for r in result if contains(r["title"], include_keyword)]
    result = [(name, r["title"], r["link"], YouTube(r["link"]).publish_date.date(), r["title"], r["link"]) for r in result]

    if len(result) == 0:
        with open("No vids found.txt", "a", encoding='utf-8') as file:
            file.write(f"{name}\n")

    return result 

charts = get_charts()[:100]
db = sqlite3.connect(VIDEO_DB_NAME, isolation_level=None)
to_skip = [r[0] for r in db.execute('''SELECT DISTINCT song_name FROM videos''')]
charts = [c for c in charts if c[0] not in to_skip]
db.close()

statement = ''' 
    INSERT INTO videos (song_name, vid_name, link, publish_date)
    SELECT ?,?,?,?
    WHERE NOT EXISTS(
        SELECT vid_name, link FROM videos WHERE
        vid_name=? AND link=?
    )
'''
def scrape(name:str, pb:tqdm):
    try:
        vids = search_video(name[0])

        db = sqlite3.connect(VIDEO_DB_NAME, isolation_level=None)
        db.executemany(statement, vids)
        db.close()
    except Exception as e:
        with open("logs/Problematic Songs.txt", "a", encoding='utf-8') as file:
            file.write(f"{name}\n")
        pb.failed += 1
        pb.set_description(f"{pb.failed} failed")
        raise e
    finally:    
        pb.update(1)

with tqdm(total=len(charts), desc="Retrieve charts") as pb:
    pb.failed = 0
    links = Parallel(-1,require="sharedmem")(delayed(scrape)(chart,pb) for chart in charts)

import re
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

Path(VID_DIR := "data/videos").mkdir(parents=True,exist_ok=True)
class Video():
    def __init__(self, name, title, link, date) -> None:
        self.name = name
        self.title = title
        self.link = link
        self.date = date

    @property
    def temp_file(self):
        return f"{VID_DIR}/temp_{sanitize(self.name)} - {sanitize(self.title)}.mp4"
    
    @property
    def mp4_file(self):
        return f"{VID_DIR}/{sanitize(self.name)} - {sanitize(self.title)}.mp4"

def sort_and_group_videos() -> list[Video]:
    statement = ''' 
        SELECT song_name, vid_name, link, publish_date from videos
        ORDER BY song_name, publish_date DESC
    '''

    db = sqlite3.connect(VIDEO_DB_NAME, isolation_level=None)
    vids = [Video(*v) for v in db.execute(statement)]
    db.close()

    def difficulty(vid:str):
        if "Re:Mas".lower() in vid.lower(): return "Re:MASTER"
        if "Mas".lower() in vid.lower(): return "MASTER"
        return "Other"
    
    groups :dict[str,list[Video]] = {}
    for vid in vids:
        groups[difficulty(vid.title)] = groups.get(difficulty(vid.title), []) + [vid]
    
    for diff, vids in groups.items():
        latest_vids = []
        for name in set(v.name for v in vids):
            vids_sorted = list(v for v in vids if v.name == name)
            latest_vids += vids_sorted[0:2]
        groups[diff] = latest_vids

    return groups["MASTER"] + groups["Re:MASTER"]

videos = sort_and_group_videos()

[os.remove(f"{VID_DIR}/{file}") for file in os.listdir(VID_DIR) if file.startswith("temp_")]
[os.remove(f"{VID_DIR}/{file}") for file in os.listdir(VID_DIR) if not file.endswith(".mp4")]

def download(vid: Video, pb:tqdm) -> None|tuple[str,str]:

    def on_progress(stream: Stream, chunk: bytes, bytes_remaining: int):
        filesize = stream.filesize
        bytes_received = filesize - bytes_remaining
        pb_sub.n = bytes_received
        pb_sub.refresh()

    try:
        vid_stream = YouTube(vid.link, on_progress_callback=on_progress).streams.get_highest_resolution()
        filesize = vid_stream.filesize
        pb_sub = tqdm(total=filesize, unit="B", unit_scale=True, desc=vid.name, leave=False)
        
        vid_stream.download(filename=vid.temp_file,skip_existing=False)
        move(vid.temp_file, vid.mp4_file)

    except Exception as e:
        with open("logs/Problematic Songs.txt", "a", encoding='utf-8') as file:
            file.write(f"{vid.name} {vid.link}      {type(e)}\n")
    finally:
        pb.update(1)

skip_download = [f"{VID_DIR}/{vid}" for vid in os.listdir(VID_DIR)]
videos = [v for v in videos if v.mp4_file not in skip_download]
with tqdm(total=len(videos), desc="Downloading videos") as pb:
    # [download(v, pb) for v in videos]
    Parallel(-1, require="sharedmem")(delayed(download)(v, pb) for v in videos)