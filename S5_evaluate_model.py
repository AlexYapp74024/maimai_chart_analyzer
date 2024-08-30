#%% Imports
import os
from itertools import chain
from pathlib import Path
import random
import shutil
import scipy.interpolate
import scipy.interpolate.interpnd
import scipy.interpolate.interpolate
import scipy.stats
import scipy.stats._stats_mstats_common
import torch
from slicerator import Slicerator
from tqdm import tqdm
import cv2
import numpy as np
from pims import Frame
import pims
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results
import supervision as sv
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track

DATA_DIR = Path("data")
MODELS_DIR = Path("runs")

S = 3180
L = S+ 100
#%% Helper funcs
import pickle
from vlc import MediaPlayer
import random
import colorsys
def ramdom_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (r,g,b)
_colors = [ramdom_color() for _ in range(256)]
def colors(i:int):
    return _colors[int(i) % len(_colors)]

def animate_images(images: list[np.ndarray], cmap = None):
    images = list(images)
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(images[0], cmap)
    def animate(k):
        im.set_array(images[k])
        return im,
    ani = animation.FuncAnimation(fig, animate, frames=len(images), blit=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.close()
    return HTML(ani.to_jshtml())

@pims.pipeline(ancestor_count = 3)
def label_results(image:Frame,res:Results, frame = 0, thickness=2, font_scale=0.8, lable_frame = True, names = None):
    if names is None: names = res.names
    img = image
    for r in res.boxes.cpu().numpy():
        if not r.is_track: continue
        p1,p2 = np.int64(np.reshape(r.xyxy,(2,2)))
        color = colors(r.id)
        img = cv2.rectangle(img,p1,p2,color,thickness)
        
        label = f"{r.id[0]}. {names[r.cls[0]]}:{r.conf[0]:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x,y = p1 - np.array([0,h*2])
        img = cv2.rectangle(img, (x, y - h), (x + w, y), color, -1)
        img = cv2.putText(img, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
                
    if lable_frame: 
        img = cv2.putText(img, f"{frame}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    return Frame(img)

@pims.pipeline(ancestor_count = 3)
def label_tracks(images:Frame,tracks:list[Track], frame = 0, thickness=2, font_scale=0.8, names = None, orig=True):
    if names is None: names = model.names
    img = images.copy()
    for t in tracks:
        wh = (t.to_tlwh(orig=orig) / 2)[2:4]
        # Somehow top left is actually midpoint
        p1,p2 = np.int64(np.reshape(t.to_tlbr(orig=orig) - np.array((*wh,*wh)),(2,2)))
        img = cv2.rectangle(img,p1,p2,colors(t.track_id),thickness)
        color = colors(int(t.det_class))

        label = f"{t.track_id}. {names[int(t.det_class)]}:{t.det_conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x,y = p1 - np.array([0,h*2])
        img = cv2.rectangle(img, (x, y - int(h*1.5)), (x + w, y), color, -1)
        img = cv2.putText(img, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        
    if frame > 0: 
        img = cv2.putText(img, f"{frame}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    return Frame(img)

@pims.pipeline
def tresh_mask_rgb(image : Frame, tresh = 240, max = 255, method = cv2.THRESH_TOZERO):
    mask = cv2.threshold(image,tresh,max,method)[1]
    return Frame(mask)

def cache_opreation(call, path:Path):
    if not path.exists(): 
        obj = call()
        MediaPlayer('notif.mp3').play()
        with open(path, "wb") as file: pickle.dump(obj, file)
    
    with open(path, "rb") as file: return pickle.load(file)

def pim_export(pipeline, file_name, rate = 60):
    pims.export(pipeline,file_name,rate=rate)
    MediaPlayer('notif.mp3').play()
#%% YOLO Detection
model_file = list(MODELS_DIR.glob("**/best.pt"))[-1]
model = YOLO(model_file, verbose=False)

video_path = (DATA_DIR/"videos/out.mp4")
video_path = list((DATA_DIR/"videos").glob("*"))[-1]
frames = pims.open(video_path.__str__())

@pims.pipeline
def predict_chunks(frame) -> list[list[Results]]:
    if len(frame) > 1: return model.track(list(frame), verbose=False, persist=True)
    return model.track(frame, verbose=False, persist=True)

def srange(obj):
    if isinstance(obj, int): return Slicerator(range(obj))
    return Slicerator(range(len(obj)))

from typing import MutableSequence
def schunks(xs, n, window = 0):
    n = max(1, n)
    return Slicerator([xs[i:i+n] for i in range(0, len(xs)-window, n-window)])

import hashlib
(TEMP_RESULT_DIR := Path("temp") / hashlib.md5(str(video_path.name).encode()).hexdigest()).mkdir(exist_ok=True,parents=True)
def strip(r:Results):
    r = r.cpu()
    r.orig_img = None
    return r

def do(): return [strip(r) for rs in tqdm(predict_chunks(schunks(frames[:-1], 32))) for r in rs]
results = cache_opreation(do, TEMP_RESULT_DIR / "res_.pyz")

labeled_results = label_results(frames[:-1],Slicerator(results),srange(len(results)))
pim_export(labeled_results,TEMP_RESULT_DIR/ "raw_res.mp4")
# animate_images(labeled_results[S:L])

#%% Tap Spots
from InvariantTM import BBox, Template_Matcher
from joblib import delayed, Parallel
def mid_point(tlbr: np.ndarray):
    return (tlbr[:2] + tlbr[2:4])/2

@pims.pipeline
def match_image(image:np.ndarray, i:int=0, templates:list[Template_Matcher] = []) -> list[BBox]:
    points_list = sum((t.match(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) for t in templates),[])
    for p in range(len(points_list)): points_list[p].frame_id = i
    rects  = [p.tlbr for p in points_list]
    scores = [p.conf for p in points_list]
    return [points_list[i] for i in cv2.dnn.NMSBoxes(rects, scores, 0.7, 0.8)]

def match_templates(images:list[np.ndarray], templates:list[Template_Matcher]) -> list[list[BBox]]:
    return Parallel(-1)(delayed(match_image)(image, i, templates) for i,image in enumerate(images))

def do():
    method=cv2.TM_CCORR_NORMED
    tap_spot_matchers  = [Template_Matcher(0,t,method,0.95, rot_range=[0,360], rot_interval=45, 
                                       scale_range=[95,105], scale_interval=5,) 
                          for t in pims.open("templates/Tap Location/*")[:1]]

    tap_spot_bboxes : list[BBox] = sum(match_templates(tqdm(tresh_mask_rgb(frames, 240)[300:3300:10]), tap_spot_matchers),[])
    rects  = [b.tlbr for b in tap_spot_bboxes]
    scores = [b.conf for b in tap_spot_bboxes]
    tap_spots : list[BBox] = [tap_spot_bboxes[i] for i in cv2.dnn.NMSBoxes(rects, scores, 0.8, 0.6)]
    tap_spots : list[np.ndarray] = [mid_point(t.tlbr) for t in tap_spots]
    return tap_spots

tap_spots = cache_opreation(do, TEMP_RESULT_DIR / "tap_spots.pyz")

from statistics import mean, mode
#%% Deep Sort
names = {4 : "Hold Touch", 3 : "Touch", 0 : "Tap", 1 : "Hold", 2 : "Star"}
remap_vals = {km : next(k for k,v in names.items() if v in vm) for km,vm in model.names.items()}

MAX_AGE = 10
MIN_AGE = 5
from copy import deepcopy
def deepsort(images: list[Frame], res: list[Results]) -> list[Track]:
    tracker = DeepSort(max_age=MAX_AGE)
    age_dict = {}

    def _next(frame:int, image:Frame , res: Results) -> list[Track]:
        detections = [[np.int64(p.xywh[0]), p.conf, p.cls] for p in res.boxes.cpu().numpy()]
        tracks : list[Track] = tracker.update_tracks(detections,frame = image, others=res)
        for t in tracks:
            if t.det_conf is None: t.det_conf = 0.3
            t.age = age_dict.get(t.track_id, 0) + 1
            t.frame = frame
            t.track_id = int(t.track_id)
            t.det_class = remap_vals[int(t.det_class)]
            age_dict[t.track_id] = t.age
        return deepcopy(tracks)
    
    tracks = [_next(f, i, r) for f, (i, r) in enumerate(zip(images, tqdm(res)))]
    too_short = [k for k,v in age_dict.items() if v < MIN_AGE]
    tracks = [[t for t in ts if t.track_id not in too_short] for ts in tracks]
    return tracks 

def do(): return deepsort(frames, results)
tracks : list[list[Track]] = cache_opreation(do, TEMP_RESULT_DIR / "ftrack.pyz")

labeled_tracks = label_tracks(frames[:-1],Slicerator(tracks),srange(len(tracks)), names = names)
# pim_export(labeled_tracks,TEMP_RESULT_DIR/ "track0.mp4",rate=30)
# animate_images(labeled_tracks[S:L])

# %% Clean Up
def dist(p1,p2): return np.linalg.norm(p1-p2)
c = sum(tap_spots) / len(tap_spots)
max_dist = int(mean([dist(c,t) for t in tap_spots]) + 10)
min_dist = int(min([dist(t0,t1) for t0 in tap_spots for t1 in tap_spots if all(t0 != t1)])) - 10

# Somehow top left midpoint
def dist_from_center(track: Track): 
    return dist(track.to_ltrb()[:2],c)
def track_dist(t0: Track, t1: Track): 
    return dist(t0.to_ltrb()[:2],t1.to_ltrb()[:2])

touches = [k for k,v in names.items() if "touch" in v.lower()]
def is_touch(track:Track): return int(track.det_class) in touches
stars   = [k for k,v in names.items() if "star" in v.lower()]
def is_star (track:Track): return int(track.det_class) in stars

def within_circle(ts: Track): 
    return dist_from_center(ts) < max_dist
def start_on_mid(ts: list[Track]): 
    return min(dist_from_center(t) for t in ts) < (max_dist // 2) or is_touch(ts[0]) or is_star(ts[0])
def ends_on_edge(ts: list[Track]): 
    return max(dist_from_center(t) for t in ts) > (max_dist *.75) or is_touch(ts[0])
def static_star(ts: list[Track]): 
    return is_star(ts[0]) and max(track_dist(ts[0], ts[-1]), track_dist(ts[0], ts[len(ts) // 2])) <= min_dist

tracks_filtered = tracks.copy()
tracks_by_id : dict[str, list[Track]] = {}
for t in chain(*tracks_filtered): tracks_by_id[t.track_id] = tracks_by_id.get(t.track_id,[]) + [t]

start_on_mids = [k for k,v in tracks_by_id.items() if not start_on_mid(v)]
ends_on_edges = [k for k,v in tracks_by_id.items() if not ends_on_edge(v)]
static_stars  = [k for k,v in tracks_by_id.items() if static_star(v)]

exclude_ids = start_on_mids + ends_on_edges + static_stars  

tracks_filtered = ([t for t in ts if within_circle(t)] for ts in tracks_filtered)
tracks_filtered = ([t for t in ts if t.track_id not in exclude_ids] for ts in tracks_filtered)
tracks_filtered = list(tracks_filtered)

labeled_tracks = label_tracks(frames[:-1],Slicerator(tracks_filtered),srange(len(tracks_filtered)), names=names)
tracks_by_id : dict[str, list[Track]] = {}
for t in chain(*tracks_filtered): tracks_by_id[t.track_id] = tracks_by_id.get(t.track_id,[]) + [t]

# animate_images(labeled_tracks[S:S+100])
# % split detections
_next_id = max(tracks_by_id.keys())
def next_id():
    global _next_id
    _next_id += 1
    return _next_id

distinct_ids = {k: v for k, ts in tracks_by_id.items() if len(v:=set(t.det_class for t in ts)) > 1}
def break_up_tracks(ts:list[Track], cs:list[int]):
    starts = sorted([next(f for f,t in enumerate(ts) if t.det_class == c) for c in cs])[1:] + [len(ts)]
    for t0,t1 in zip(starts[:-1], starts[1:]):
        _next = next_id()
        for t in ts[t0:t1]: t.track_id = _next

for k,v in distinct_ids.items(): break_up_tracks(tracks_by_id[k],v)

tracks_by_id : dict[str, list[Track]] = {}
for t in chain(*tracks_filtered): tracks_by_id[t.track_id] = tracks_by_id.get(t.track_id,[]) + [t]

too_short = [k for k,v in tracks_by_id.items() if len(v) < 15 and v[0].det_class != 3]
tracks_filtered = ([t for t in ts if t.track_id not in too_short] for ts in tracks_filtered)
tracks_filtered = [t if i > 310 else [] for i, t in enumerate(tracks_filtered)]

# %% 
S = 1100
labeled_tracks = label_tracks(frames[:-1],Slicerator(tracks_filtered),srange(len(tracks_filtered)), names=names, orig=True)
animate_images(labeled_tracks[S:S+100])
# pim_export(labeled_tracks,TEMP_RESULT_DIR/ "track1.mp4",rate=30)

# %% create datasets
def track_2_coco(t:Track, ih, iw):
    x,y,w,h = t.to_ltwh(orig=True)
    return f"{int(t.det_class)} {x/iw} {y/ih} {w/iw} {h/ih}"

(DATA_DIR / "images").mkdir(exist_ok=True)
(DATA_DIR / "labels").mkdir(exist_ok=True)

digits = len(str(len(tracks_filtered)))
@pims.pipeline(ancestor_count = 2)
def copy_images(image:Frame, frame):
    filename = f"{(TEMP_RESULT_DIR.name)}---{str(frame).zfill(digits)}"
    cv2.imwrite((DATA_DIR / "images" / filename).with_suffix(".webp"),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

@pims.pipeline(ancestor_count = 3)
def create_coco_dataset(image:Frame, tracks:list[Track], frame):
    filename = f"{(TEMP_RESULT_DIR.name)}---{str(frame).zfill(digits)}"
    ih, iw, _ = image.shape
    with open((DATA_DIR / "labels" / filename).with_suffix(".txt"), "w") as file: file.write("\n".join(track_2_coco(t, ih, iw) for t in tracks))

# dataset = create_coco_dataset(tresh_mask_rgb(frames[:-1]), Slicerator(tracks_filtered), srange(len(tracks_filtered)))[::5]
dataset = copy_images(tresh_mask_rgb(frames[:-1],128,method = cv2.THRESH_TOZERO), srange(len(tracks_filtered)))[::5]

def do(d, pb:tqdm): d, pb.update(1)
with tqdm(total = len(dataset)) as pb:
    Parallel(-1, require="sharedmem")(delayed(do)(d, pb) for d in dataset)

MediaPlayer('notif.mp3').play()
pass
# %% create obb and mask datasets
S = 5850
animate_images(tresh_mask_rgb(frames[:-1],128,method = cv2.THRESH_TOZERO)[S:S+100])