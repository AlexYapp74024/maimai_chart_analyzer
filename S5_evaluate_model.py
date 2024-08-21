#%%
import os
from pathlib import Path
import random
import shutil
import yaml
import cv2
import numpy as np

DATASET_DIR = Path("data")
MODELS_DIR = Path("runs")

#%%
model_files = [Path(root) / file for root, _, files in os.walk(MODELS_DIR) 
               for file in files if file.endswith(".pt")]
model_files

#%%
from ultralytics import YOLO
from ultralytics.engine.results import Results

model = YOLO(model_files[-2])
model.info()

# %%
images = os.listdir(DATASET_DIR/"images")
images
image = cv2.imread(DATASET_DIR/"images"/images[302])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = model(image, show=True)
cv2.waitKey(0)
results[0].boxes

# %%
xyxys = results[0].boxes.xywhn
xyxys

# %%
im = image.copy()
for xyxy in xyxys:
    xy1 = xyxy[0:2].to('cpu').numpy().astype(np.int64)
    xy2 = xyxy[2:4].to('cpu').numpy().astype(np.int64)
    im = cv2.rectangle(im, xy1, xy2, (255,255,0))

# %%
# cv2.imshow("image", im)
# cv2.waitKey(0)

# %%
p = Path("data/videos")
video = list(p/f for f in os.listdir(p))[0]
video

# %%
results = model(video, save=True)
results

# %%
