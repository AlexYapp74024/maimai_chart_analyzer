# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pathlib import Path
from joblib import delayed, Parallel
from time import perf_counter
from tqdm import tqdm 
# from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from IPython.display import HTML
from InvariantTM import BBox, Template_Matcher
import itertools
import shutil

import random
import colorsys
def ramdom_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (r,g,b)

colors = [ramdom_color() for _ in range(50)]

SCALE = 1.0
def load_image_rgba(path:Path):
    image = cv2.cvtColor(cv2.imread(path,cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA)
    h,w = np.array(image.shape[:2]) * SCALE
    return cv2.resize(image, (int(w), int(h)))

def load_image(path:Path):
    image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    h,w = np.array(image.shape[:2]) * SCALE
    return cv2.resize(image, (int(w), int(h)))

# %%
start = 6700
start = 4000
length = 80
work_paths = list(Path("data/images").glob("*"))[start:start+length]
image_list = [load_image(p) for p in work_paths]

def animate_images(images: list[np.ndarray], cmap = None):
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

def draw_bboxes(image_list: list[np.ndarray], bboxes : list[list[BBox]], scale = 1.0, thickness = 2, font_scale = 0.3):
    images = []
    for image, bboxs in zip(image_list, bboxes):
        h,w = np.array(image.shape[:2]) * scale
        img = cv2.resize(image, (int(w), int(h)))
        for b in bboxs:
            track_id = b.track_id if b.track_id else 0
            color = colors[track_id % len(colors)]
            tlbr = (b.tlbr*scale).astype(np.int32)
            img = cv2.rectangle(img,tlbr[:2], tlbr[2:],color, thickness)
            
            label = f"{b.track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x,y = tlbr[:2] - np.array([0,h])
            img = cv2.rectangle(img, (x, y - h), (x + w, y), color, -1)
            img = cv2.putText(img, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        images.append(img)
    return images

# animate_images(image_list)
# %%
method=cv2.TM_CCORR_NORMED
matched_thresh=0.85
TEMPLATE_DIR = Path("templates")

tap_templates  = [load_image_rgba(i) for i in (TEMPLATE_DIR / "Tap").glob("*")]
template_matchers = [Template_Matcher(t,method,matched_thresh,
                                      rot_range=[22,90+22], rot_interval=45, 
                                      scale_range=[70,111], scale_interval=5,) for t in tap_templates]

tap_spot_templates  = [load_image_rgba(i) for i in (TEMPLATE_DIR / "Tap Location").glob("*")]
tap_spot_matchers = [Template_Matcher(t,method,0.95,
                                      rot_range=[0,360], rot_interval=45, 
                                      scale_range=[95,105], scale_interval=5,) for t in tap_spot_templates]

crit_templates = [load_image_rgba(i) for i in (TEMPLATE_DIR / "Crit Perf").glob("*")]
crit_matchers = [Template_Matcher(t,method,matched_thresh,
                                  rot_range=[22,360+22], rot_interval=45, 
                                  scale_range=[80,111], scale_interval=5,) for t in crit_templates]

def match_templates(images:list[np.ndarray], templates:list[Template_Matcher]) -> list[list[BBox]]:
    def _match_image(i:int, image:np.ndarray, templates:list[Template_Matcher]) -> list[BBox]:
        points_list = sum((t.match(image) for t in templates),[])
        for p in range(len(points_list)): points_list[p].frame_id = i
        rects  = [p.tlbr for p in points_list]
        scores = [p.conf for p in points_list]
        return [points_list[i] for i in cv2.dnn.NMSBoxes(rects, scores, 0.7, 0.8)]
    return Parallel(-1)(delayed(_match_image)(i,cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), templates) for i,image in enumerate(images))
    return list(_match_image(i,cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), templates) for i,image in enumerate(images))

t = perf_counter()
tap_spot_bboxes : list[BBox] = sum(match_templates(tqdm(image_list), tap_spot_matchers),[])
print(f"{perf_counter() - t:.3f}")

t = perf_counter()
bboxes_list = match_templates(tqdm(image_list), template_matchers)
print(f"{perf_counter() - t:.3f}")
# %%
rects  = [b.tlbr for b in tap_spot_bboxes]
scores = [b.conf for b in tap_spot_bboxes]
tap_spots : list[BBox] = [tap_spot_bboxes[i] for i in cv2.dnn.NMSBoxes(rects, scores, 0.7, 0.8)]
# %%
MAX_AGE = 15

def deepsort(image_list:list[np.ndarray], point_list: list[list[BBox]], reverse = False):
    def track_2_bbox(track:Track) -> BBox:
        conf = track.det_conf if track.det_conf else 0.3
        # other:BBox = track.others
        return BBox(track.to_ltrb(),
                    conf=conf,
                    angle=0,
                    track_id=int(track.track_id))

    tracker = DeepSort(max_age=MAX_AGE)
    bboxes: list[list[tuple[np.ndarray, int]]] = []
    iters = list(zip(image_list, point_list))
    if reverse: iters = reversed(iters)
    for image, points in iters:
        tracks: list[Track] = tracker.update_tracks([[p.tlwh(), p.conf] for p in points],frame = image,others=points)
        bboxes.append([track_2_bbox(track) for track in tracks])
    return bboxes

def remove_dropped(bboxes:list[list[BBox]]):
    past_tids = set(b.track_id for b in bboxes[MAX_AGE-1])
    for t, bboxs in enumerate(bboxes[MAX_AGE:],MAX_AGE): 
        tids = set(b.track_id for b in bboxs)
        dropped = past_tids - tids
        for i in range(t-MAX_AGE,t):
            bboxes[i] = [b for b in bboxes[i] if b.track_id not in dropped]
        past_tids = tids
    return bboxes

#%%
bboxes = bboxes_list
bboxes = deepsort(image_list, bboxes, reverse = True)
bboxes = remove_dropped(bboxes)
bboxes = deepsort(image_list, bboxes)
bboxes : list[list[BBox]] = list(reversed(remove_dropped(bboxes)))
for frame, boxes in enumerate(bboxes):
    for i, b in enumerate(boxes):
        bboxes[frame][i].frame_id = frame

box_by_track_ids : dict[int, list[BBox]]= {}
filtered_conf = [[b for b in boxs if b.conf >= 0.6] for boxs in bboxes]
for box in sum(filtered_conf,[]):
    box_by_track_ids[box.track_id] = box_by_track_ids.get(box.track_id, []) + [box]

min_max_frames = {t:(min(boxes,key=lambda b:b.frame_id).frame_id, max(boxes,key=lambda b:b.frame_id).frame_id) for t,boxes in box_by_track_ids.items()}
bboxes = [[b for b in boxs if b.conf >= 0.6 or (b.frame_id >= min_max_frames[b.track_id][0] and b.frame_id <= min_max_frames[b.track_id][1])] 
          for boxs in bboxes]
# animate_images(draw_bboxes(image_list,bboxes,scale=0.5,thickness=2,font_scale=0.4))
#%%
def is_outlier(data, m=0.5): return abs(data - np.median(data)) > m * np.std(data)

nearest_to_taps : list[BBox] = [max(box_, key = lambda b: b.frame_id) for box_ in box_by_track_ids.values()]
tap_sensors = [b.center() for b in tap_spot_bboxes]
distances = np.array([min(np.sum(np.square(fb.center() - ts)) for ts in tap_sensors) for fb in nearest_to_taps])
outliers_idx : list[int] = [box.track_id for box, reject in zip(nearest_to_taps, is_outlier(distances)) if reject]

box_by_track_ids : dict[int, list[BBox]]= {}
for box in sum(filtered_conf,[]):
    if box.track_id in outliers_idx : continue
    box_by_track_ids[box.track_id] = box_by_track_ids.get(box.track_id, []) + [box]

bboxes : list[list[BBox]] = [[]] * len(bboxes)
for b in sum(list(box_by_track_ids.values()),[]):
    bboxes[b.frame_id] = bboxes[b.frame_id] + [b]

box_by_frame_ids : dict[int, list[BBox]]= {}
for box in sum(bboxes,[]):
    box_by_frame_ids[box.frame_id] = box_by_frame_ids.get(box.frame_id, []) + [box]
#%%
frame_tracks = sorted([(k,set(v.track_id for v in vs)) for k,vs in box_by_frame_ids.items()], key = lambda kv: -len(kv[1]))

still_untracked = set(b.track_id for b in sum(bboxes,[]))
batch_frames = []
LOOK_AHEAD = 20
while len(still_untracked) > 0:
    look_aheads = []
    for frame, boxes in frame_tracks[0:LOOK_AHEAD]:
        if len(boxes) != len(frame_tracks[0][1]): break
        look_aheads.append((frame, boxes))
    
    frame, boxes = look_aheads[len(look_aheads)//2]
    batch_frames.append(frame)
    [still_untracked.remove(b) for b in boxes if b in still_untracked]
    frame_tracks = sorted([(k,set(v.track_id for v in vs if v.track_id in still_untracked)) for k,vs in box_by_frame_ids.items()], key = lambda kv: -len(kv[1]))

batch_boxes = {i: box_by_frame_ids[i] for i in batch_frames}
output_boxes : list[list[BBox]] = [[]] * len(bboxes_list)
for b in sum(batch_boxes.values(),[]):
    output_boxes[b.frame_id] = output_boxes[b.frame_id] + [b]
batch_frames
# %%
animate_images(draw_bboxes(image_list,output_boxes,scale=0.5, font_scale=0.4))
# %%
import torch
if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
# device = torch.device("cpu")

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

sam2_checkpoint = "model_data/sam2_hiera_base_plus.pt"
model_cfg = "sam2_hiera_b+.yaml"
shutil.rmtree(video_forward_dir := Path("temp_forward"),ignore_errors=True), video_forward_dir.mkdir(exist_ok=True)
TOTAL_FRAMES = len(work_paths)
for i,p in enumerate(work_paths):
    cv2.imwrite((video_forward_dir/f"{i:05d}").with_suffix(".jpg"),cv2.imread(p,cv2.IMREAD_UNCHANGED))

predictor : SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
forward_inference_state = predictor.init_state(video_path=video_forward_dir.__str__())
# reverse_inference_state = predictor.init_state(video_path=video_reverse_dir.__str__())
# %%
def initial_points(predictor:SAM2VideoPredictor, inference_state:dict, bboxes: list[BBox], frame_id:int):
    predictor.reset_state(inference_state)
    current_track_id = 0
    outs=[]
    for box in bboxes:
        if box.track_id is None:
            track_id = current_track_id
            current_track_id += 1
        else:
            track_id = box.track_id
        outs.append(predictor.add_new_points_or_box(inference_state=inference_state,frame_idx=frame_id,obj_id=track_id,box=box.tlbr)) 
            # a, track_ids, masks = predictor.add_new_points_or_box(inference_state=inference_state,frame_idx=frame_idx,obj_id=track_id,box=bbox.tlbr)
    # a, track_ids, masks = list(zip(*outs))
    # return a, track_ids, masks

def segment(predictor:SAM2VideoPredictor, inference_state:dict, bboxes: list[BBox], reverse: bool = False) -> dict[int,dict[int,np.ndarray]]:
    video_segments = {}  # video_segments contains the per-frame segmentation results
    n_distinct_objects = len(set(b.track_id for b in bboxes))
    distinct_objects = set()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=reverse):
        dicts_ = {
            out_obj_id: (out_mask_logits[i] > 0.0)
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        dicts_ = {k:v.cpu().numpy() for k,v in dicts_.items() if torch.any(v)}
        if len(dicts_):
            video_segments[out_frame_idx] = dicts_
            for o in out_obj_ids: distinct_objects.add(o)
        elif len(video_segments) >= n_distinct_objects: 
            break
        
    return video_segments

def draw_mask(image: np.ndarray, masks: list[np.ndarray], indices:list[int], weight = 0.7):
    masked = image.copy()
    for mask, idx in zip(masks, indices):
        for (x,y) in zip(*np.where(mask > 0.0)):
            masked[x,y] = masked[x,y] * (1 - weight) + np.array(colors[idx]) * weight
    return masked

def mask_segments(video_segments:dict, image_list:list):
    masked_img = []
    for i, img in tqdm(enumerate(image_list)):
        if i not in video_segments.keys(): 
            masked_img.append(img)
            continue
        obj_ids, masks = list(zip(*video_segments[i].items()))
        masks = [m[0] for m in masks]
        masked_img.append(draw_mask(img,masks,obj_ids))
    return masked_img

def mask_2_tlbr(mask):
    x,y,w,h = cv2.boundingRect(cv2.findNonZero(mask[0].astype(np.int8)))
    return np.array((x,y,x+w,y+h))

def merge_masks_segments(frame:int, boxs:list[BBox]) -> list[BBox]:
    initial_points(predictor, forward_inference_state, boxs, frame)
    reverse_video_segments = segment(predictor, forward_inference_state, boxs)
    forward_video_segments = segment(predictor, forward_inference_state, boxs, True)

    video_segments = forward_video_segments.copy()
    for frame, vals in reverse_video_segments.items():
        if video_segments.get(frame, None) is None:
            video_segments[frame] = vals.copy()
            continue
        for track_id, mask in vals.items():
            if video_segments[frame].get(track_id, None) is None:
                video_segments[frame][track_id] = mask
            else:
                video_segments[frame][track_id] += mask
    
    return [BBox(mask_2_tlbr(mask),0,0.5,frame,track_id) for frame, vals in video_segments.items() for track_id, mask, in vals.items()]

# SAM_boxes = merge_masks_segments(frame, boxs)
# SAM_boxes = sum([merge_masks_segments(frame, boxs) for frame, boxs in batch_boxes.items()], [])
# output_boxes : list[list[BBox]] = [[]] * len(bboxes)
# for b in SAM_boxes:
#     output_boxes[b.frame_id] = output_boxes[b.frame_id] + [b]
# %%
animate_images(draw_bboxes(image_list,output_boxes,scale=0.5,thickness=2,font_scale=0.4))
# output_boxes
# %%
