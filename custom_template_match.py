# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pathlib import Path
from joblib import delayed, Parallel
import cProfile, pstats
from time import perf_counter
from tqdm import tqdm 
# from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from IPython.display import HTML
from InvariantTM import BBox, Template_Matcher
import numba as nb

@nb.njit
def sum_2d(arr:np.ndarray):
    return np.sum(np.sum(arr,-1),-1)

def match_TM_CCORR_NORMED(images:list[np.ndarray], template:np.ndarray):
    @nb.njit
    def _match(image_stack:np.ndarray, template:np.ndarray, out_array:np.ndarray):
        norm_T = np.sum(np.square(template[0]))
        th,tw = template.shape[1:3]
        for y in nb.prange(h):
            for x in nb.prange(w):
                img = image_stack[:,y:y+th,x:x+tw]
                img_sq = sum_2d(np.square(img))
                result = sum_2d(img * template)
                normalize = np.sqrt(norm_T * img_sq) + 0.000001
                out_array[:,y,x] = result / normalize
        return out_array
    
    n = len(images)
    image_stack = np.array(images).astype(np.float32)
    template = np.array([template] * n).astype(np.float32)
    h,w   = np.array(image_stack.shape[1:3]) - np.array(template.shape[1:3])
    matches = _match(image_stack,template,np.full((n,h,w), 0.0))
    return matches

t = perf_counter()
test_lists = image_list[:5]
cv_matches = [cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),cv2.cvtColor(tap_templates[0], cv2.COLOR_RGB2GRAY),method) for image in test_lists]
print(f"{perf_counter() - t:.3f}")

t = perf_counter()
image_list_gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in test_lists]
matches = match_TM_CCORR_NORMED(image_list_gray,cv2.cvtColor(tap_templates[0], cv2.COLOR_RGB2GRAY))
print(f"{perf_counter() - t:.3f}")

# np.where(cv_matches >= 0.85)
# %%

# animate_images(draw_bboxes(image_list, bboxes, scale = 0.5))

# %%
y,x = 79,496

cv_matches  = [cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                                 cv2.cvtColor(tap_templates[0], cv2.COLOR_RGB2GRAY),
                                 cv2.TM_CCORR_NORMED)[y,x]
               for image in image_list]
image_list_gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in image_list[:10]]
image_stack = np.array(image_list_gray).astype(np.float32)
template = np.array([cv2.cvtColor(tap_templates[0], cv2.COLOR_RGB2GRAY)] * len(image_list_gray)).astype(np.float32)

norm_T = np.sum(np.square(template[0]))
th,tw = template.shape[1:3]
img = image_stack[:,y:y+th,x:x+tw]
img_sq = np.square(img)
img_sq = np.sum(img_sq,axis=-1)
img_sq = np.sum(img_sq,axis=-1)

result = img * template
result = np.sum(result,-1)
result = np.sum(result,-1)

normalize = np.sqrt(norm_T * img_sq) + 0.000001
result /= normalize

(result - cv_matches)
# %%
# np.unique(image_stack[0,x:x+th,y:y+tw] == image_list_gray[0][x:x+th,y:y+tw])