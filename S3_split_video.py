#%%
import os
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import re
import hashlib
from pathlib import Path
import itertools
from dataclasses import dataclass

VID_DIR = Path("data/videos")
vids = list(VID_DIR.glob("*"))

Path(IMG_DIR := "data/images").mkdir(exist_ok=True, parents=True)
Path(LOG_DIR := "logs").mkdir(exist_ok=True, parents=True)
FRAME = 1
# SCALE = 2

OUT_FORMAT = ".webp"

from matplotlib import pyplot as plt
FILES_INDEX = (file for file in os.listdir(IMG_DIR) if file.endswith(OUT_FORMAT))
FILES_INDEX = (re.split(r"---|\.",file) for file in FILES_INDEX)
FILES_INDEX = list((ids[0] , int(ids[1])) for ids in FILES_INDEX)
EXISTING_INDEX = {}
for vid, fid in FILES_INDEX: 
    EXISTING_INDEX[vid] = EXISTING_INDEX.get(vid,[]) + [fid]

@dataclass
class SplitJob():
    frameIndex : int
    digits: int
    vidHash: str
    vidPath: Path

    def save_frame(self):
        try:
            image = iio.imread(self.vidPath,index=self.frameIndex)
            file_name = f"{IMG_DIR}/{self.vidHash}---{str(self.frameIndex).zfill(self.digits)}{OUT_FORMAT}"
            iio.imwrite(file_name, image)
        except StopIteration as e:
            pass
        except Exception as e:
            with open("logs/extract problem.txt", "w", encoding='utf-8') as file:
                file.write(f"{self.vidPath}\n")


def split_video(vidHash:str, vidPath:Path):
    meta = iio.immeta(vidPath)
    total_frames = int(meta["duration"] * meta["fps"] +0.5)

    digits = len(str(total_frames))
    skip_id = EXISTING_INDEX.get(vidHash,[])
    
    indices = list(fi for fi in range(0,total_frames,FRAME) if fi not in skip_id)

    return list(SplitJob(frame, digits, vidHash, vidPath) for frame in tqdm(indices, leave=False))

def exec(pb:tqdm, s:SplitJob):
    s.save_frame()
    pb.update(1)

hashes = list(hashlib.md5(str(v.name).encode()).hexdigest() for v in vids)
assert len(set(hashes)) == len(vids)

splitjobs = sum([split_video(*vi) for vi in zip(hashes, vids)],[])

with tqdm(total=len(splitjobs), desc="Splitting videos") as pb:
    # list((exec)(pb, s) for s in splitjobs)
    Parallel(8, require="sharedmem")(delayed(exec)(pb, s) for s in splitjobs)
# %%
