from __future__ import annotations
import cv2
import numpy as np
from numba import njit, jit, prange
from dataclasses import dataclass

box_points = []
button_down = False

class MethodError(Exception):pass

@dataclass
class BBox():
    tlbr: np.ndarray
    angle: int
    scale: int
    conf : float
    frame_id: int|None = None
    track_id: int|None = None
    class_id: int|None = None

    def tlwh(self): 
        tl, br = self.tlbr.reshape((2,2))
        return np.array((*tl, *(br - tl)))

    def center(self) -> np.ndarray: 
        tl, br = self.tlbr.reshape((2,2))
        return ((tl + br) // 2).astype(np.int32)

def rotate_image(image:np.ndarray, angle:int):
    if angle == 0: return image
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image:np.ndarray, percent):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return result

MAX_SIZE_FOR_TEMPLATE_LIST = 50

class Template_Matcher():
    @staticmethod
    def combine(matchers :list[Template_Matcher]) -> Template_Matcher:
        assert all(m.cache for m in matchers)
        assert all(m.method == matchers[0].method for m in matchers)
        assert all(m.class_id == matchers[0].class_id for m in matchers)
        assert all(m.matched_thresh == matchers[0].matched_thresh for m in matchers)

        templates = sum((m.template_list for m in matchers), [])
        out_matcher = matchers[0]
        out_matcher.template_list = templates
        return out_matcher

    def __init__(self, class_id:int, rgbtemplate:np.ndarray, method:int, matched_thresh:float, rot_range:list[int], rot_interval:int, scale_range:list[int], scale_interval:int):
        """
        rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
        method: [String] Parameter specifying the comparison method
        matched_thresh: [Float] Setting threshold of matched results(0~1).
        rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
        rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
        scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
        scale_interval: [Integer] Interval of traversing the range of scaling in percentage.
        """
        self.matched_thresh = matched_thresh
        self.method = method
        self.class_id = class_id
        self.template_gray = cv2.cvtColor(rgbtemplate, cv2.COLOR_RGB2GRAY)

        match method:
            case cv2.TM_CCOEFF | cv2.TM_CCORR | cv2.TM_SQDIFF:
                self.normalized = False
            case cv2.TM_CCOEFF_NORMED | cv2.TM_CCORR_NORMED | cv2.TM_SQDIFF_NORMED:
                self.normalized = True

        match method:
            case cv2.TM_CCOEFF | cv2.TM_CCOEFF_NORMED | cv2.TM_CCORR | cv2.TM_CCORR_NORMED:
                self.pad_value = 0.0
            case cv2.TM_SQDIFF | cv2.TM_SQDIFF_NORMED:
                self.pad_value = 1.0

        angles = list(range(rot_range[0], rot_range[1], rot_interval))
        scales = list(range(scale_range[0], scale_range[1], scale_interval))
        self.cache = (len(angles) * len(scales) <= MAX_SIZE_FOR_TEMPLATE_LIST)

        if self.cache:
            self.template_list : list[tuple[int,np.ndarray]] = []
            for angle in angles:
                for scale in scales:
                    rotated_template = rotate_image(scale_image(self.template_gray, scale), angle)
                    self.template_list.append((angle, scale, rotated_template))
        else:
            self.angles = angles
            self.scales = scales

    def match(self, img_gray:np.ndarray) -> list[BBox]:
        if self.cache:
            return self._match_cached(img_gray)
        else:
            return self._match_loop(img_gray)

    @staticmethod
    @njit
    def _process_matches(matched_points,normalized,method,matched_thresh):        
        if not normalized:
            matched_points = matched_points / (np.max(matched_points) - np.min(matched_points))

        match method:
            case cv2.TM_CCOEFF | cv2.TM_CCOEFF_NORMED | cv2.TM_CCORR | cv2.TM_CCORR_NORMED:
                satisfied_points = np.where(matched_points >= matched_thresh)
            case cv2.TM_SQDIFF | cv2.TM_SQDIFF_NORMED:
                satisfied_points = np.where(matched_points <= matched_thresh)
        return satisfied_points
              
    def _match_loop(self, img_gray:np.ndarray) -> list[BBox]:
        all_points :list[BBox] = []
        for scale in self.scales:
            for angle in self.angles:
                template = rotate_image(scale_image(self.template_gray, scale), angle)
                matched_points : np.ndarray = cv2.matchTemplate(img_gray,template,self.method) 
                satisfied_points = self._process_matches(matched_points,self.normalized,self.method,self.matched_thresh)

                h,w = template.shape[:2]
                for y,x in zip(*satisfied_points):
                    all_points.append(BBox(np.array((x,y,x+w,y+h)),angle=angle,scale=scale,
                                           conf=matched_points[y,x],class_id=self.class_id))

        return all_points
    
    def _match_cached(self, img_gray:np.ndarray) -> list[BBox]:
        matched_points : list[np.ndarray] = [cv2.matchTemplate(img_gray,template,self.method) for _,_,template in self.template_list]
        max_y, max_x = np.max([m.shape for m in matched_points], axis=0)
        matched_points = np.array([np.pad(m,((0,max_y-m.shape[0]),(0,max_x-m.shape[1])),
                                        constant_values=self.pad_value) 
                                        for m in matched_points])
        satisfied_points = self._process_matches(matched_points,self.normalized,self.method,self.matched_thresh)

        all_points :list[BBox] = []
        for tid, y,x in zip(*satisfied_points):
            angle, scale, template = self.template_list[tid]
            h,w = template.shape[:2]
            all_points.append(BBox(np.array((x,y,x+w,y+h)),angle,scale,matched_points[tid, y,x],class_id=self.class_id))

        return all_points