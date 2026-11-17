import os
import time
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from PIL import Image

from fruit_analyzer.models.classifier import FruitClassifier

def find_candidate_boxes(img: Image.Image, min_area_ratio: float = 0.02) -> List[Tuple[int, int, int, int]]:
    orig_w, orig_h = img.size
    max_dim = 640
    scale = 1.0
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        sw, sh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
        small = img.resize((sw, sh), Image.Resampling.BILINEAR)
    else:
        sw, sh = orig_w, orig_h
        small = img

    inv_scale = 1.0 / scale
    arr = np.array(small.convert("RGB"))

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # Border sampling
    borders = np.concatenate([
        arr[:6, :, :].reshape(-1, 3),
        arr[-6:, :, :].reshape(-1, 3),
        arr[:, :6, :].reshape(-1, 3),
        arr[:, -6:, :].reshape(-1, 3)
    ], axis=0)
    bg_rgb = np.median(borders, axis=0)
    color_diff = np.linalg.norm(arr.astype(float) - bg_rgb, axis=2)

    a_ch = lab[:, :, 1].astype(float) - 128.0
    b_ch = lab[:, :, 2].astype(float) - 128.0
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)
    sat = hsv[:, :, 1].astype(float)
    val = hsv[:, :, 2].astype(float)

    is_fg = (
        ((color_diff > 20.0) & (sat > 25.0) & (val > 25.0)) |
        (chroma > 18.0) |
        ((sat > 45.0) & (val > 35.0))
    ).astype(np.uint8) * 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(is_fg, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (sw * sh) * min_area_ratio

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(c)

        # Skip full-frame contours (image boundary artifacts)
        if (w_box >= 0.92 * sw and h_box >= 0.92 * sh) or area > (sw * sh * 0.85):
            continue

        # Propose primary unified contour bounding box (with slight 4% padding)
        pad_x = max(2, int(w_box * 0.04))
        pad_y = max(2, int(h_box * 0.04))
        bx1 = max(0, int((x - pad_x) * inv_scale))
        by1 = max(0, int((y - pad_y) * inv_scale))
        bx2 = min(orig_w, int((x + w_box + pad_x) * inv_scale))
        by2 = min(orig_h, int((y + h_box + pad_y) * inv_scale))
        boxes.append((bx1, by1, bx2, by2))

        # Check for multiple peaks in distance transform (only if contour is large or elongated)
        sub_mask = opened[y:y+h_box, x:x+w_box]
        dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()

        if max_dist > 14.0 and (max(w_box, h_box) > 1.4 * min(w_box, h_box) or area > (sw * sh * 0.08)):
            dilated = cv2.dilate(dist, np.ones((13, 13), np.uint8))
            peaks = (dist == dilated) & (dist > 0.45 * max_dist)
            peak_indices = np.argwhere(peaks)
            if len(peak_indices) >= 2:
                # Filter peaks to only those separated by significant distance
                selected_peaks = []
                for py, px in peak_indices:
                    too_close = False
                    for spy, spx in selected_peaks:
                        if np.sqrt((py - spy)**2 + (px - spx)**2) < (0.28 * max(w_box, h_box)):
                            too_close = True
                            break
                    if not too_close:
                        selected_peaks.append((py, px))

                if len(selected_peaks) >= 2:
                    for py, px in selected_peaks:
                        r = int(dist[py, px] * 1.15)
                        sub_x1 = max(x, x + px - r)
                        sub_y1 = max(y, y + py - r)
                        sub_x2 = min(x + w_box, x + px + r)
                        sub_y2 = min(y + h_box, y + py + r)
                        px1 = max(0, int((sub_x1 - 2) * inv_scale))
                        py1 = max(0, int((sub_y1 - 2) * inv_scale))
                        px2 = min(orig_w, int((sub_x2 + 2) * inv_scale))
                        py2 = min(orig_h, int((sub_y2 + 2) * inv_scale))
                        if (px2 - px1) > 35 and (py2 - py1) > 35:
                            boxes.append((px1, py1, px2, py2))

    if not boxes:
        boxes.append((0, 0, orig_w, orig_h))

    return boxes

def iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

def iom(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / min(a1, a2) if min(a1, a2) > 0 else 0

def detect_fruits(img: Image.Image, clf: FruitClassifier):
    orig_w, orig_h = img.size
    boxes = find_candidate_boxes(img)
    
    candidates = []
    for b in boxes:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        area = bw * bh
        if bw < 30 or bh < 30:
            continue
        crop = img.crop(b)
        pred = clf.predict(crop)
        if pred.confidence >= 0.38 and pred.top_fruit != "Unknown / Background":
            candidates.append((b, pred.top_fruit, pred.confidence, area))

    # Sort by confidence * (area ** 0.25)
    candidates.sort(key=lambda c: c[2] * (c[3] ** 0.25), reverse=True)

    final = []
    for b, fruit, conf, area in candidates:
        suppress = False
        for fb, ff, fc, fa in final:
            if iou(b, fb) > 0.32 or iom(b, fb) > 0.40:
                suppress = True
                break
        if not suppress:
            final.append((b, fruit, conf, area))

    if not final:
        full_pred = clf.predict(img)
        final.append(((0, 0, orig_w, orig_h), full_pred.top_fruit, full_pred.confidence, orig_w * orig_h))

    # Check for single fruit dominance
    if len(final) > 1:
        primary = final[0]
        prim_area = primary[3]
        if prim_area > (orig_w * orig_h * 0.40) and primary[2] > 0.85:
            all_fragments = all(iom(f[0], primary[0]) > 0.20 for f in final[1:])
            if all_fragments:
                final = [primary]

    final.sort(key=lambda d: d[0][0])
    return final

if __name__ == "__main__":
    clf = FruitClassifier()
    test_files = [
        "fruit_analyzer/data/sample_images/apple.jpg",
        "fruit_analyzer/data/sample_images/banana.jpg",
        "fruit_analyzer/data/sample_images/orange.jpg",
        "fruit_analyzer/data/sample_images/strawberry.jpg",
        "fruit_analyzer/data/sample_images/watermelon.jpg",
        "fruit_analyzer/data/sample_images/kiwi.jpg",
        "fruit_analyzer/data/sample_images/two_bananas_one_apple.jpg",
        "fruit_analyzer/data/sample_images/mixed_fruits.jpg",
        "fruit_analyzer/data/fruits_dataset_expanded/multi_fruit_scenes/apples_pears.jpg",
        "fruit_analyzer/data/fruits_dataset_expanded/multi_fruit_scenes/apple_grape.jpg",
        "fruit_analyzer/data/fruits_dataset_expanded/multi_fruit_scenes/apple_pear.jpg",
    ]
    for p in test_files:
        img = Image.open(p)
        t0 = time.time()
        res = detect_fruits(img, clf)
        dt = (time.time() - t0) * 1000.0
        print(f"{Path(p).name} -> {len(res)} items in {dt:.1f}ms: {[(r[1], f'{r[2]*100:.1f}%') for r in res]}")
