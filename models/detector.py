"""
Multi-Fruit Spatial Detector & Auto-Counter
High-performance OpenCV morphological contour extraction, distance-transform peak separation,
deep MobileNetV2 ONNX scoring, and universal geometric Non-Maximum Suppression.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

try:
    import cv2
except ImportError:
    venv_site = str(Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages")
    if venv_site not in sys.path and Path(venv_site).exists():
        sys.path.insert(0, venv_site)
    import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .classifier import FruitClassifier


@dataclass
class DetectedFruitInstance:
    instance_id: int
    fruit_name: str
    confidence: float
    box: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax (pixel coordinates)
    normalized_box: Tuple[float, float, float, float]  # 0.0 - 1.0
    color_hex: str = "#10B981"
    color_rgb: Tuple[int, int, int] = (16, 185, 129)


@dataclass
class MultiFruitDetectionResult:
    instances: List[DetectedFruitInstance]
    counts_by_fruit: Dict[str, int]
    annotated_image: Image.Image
    total_detected: int


class MultiFruitDetector:
    """
    Multi-fruit spatial segmentation, classification, and counting pipeline.
    Combines OpenCV morphological gradient saliency and distance-transform peak splitting
    with deep MobileNetV2 ONNX inference and universal non-maximum suppression.
    """

    DEFAULT_PALETTE = [
        (239, 68, 68),   # #EF4444 Red
        (234, 179, 8),   # #EAB308 Yellow
        (249, 115, 22),  # #F97316 Orange
        (34, 197, 94),   # #22C55E Green
        (168, 85, 247),  # #A855F7 Purple
        (59, 130, 246),  # #3B82F6 Blue
        (236, 72, 153),  # #EC4899 Pink
        (20, 184, 166),  # #14B8A6 Teal
    ]

    FRUIT_COLOR_MAP = {
        "Apple": ((239, 68, 68), "#EF4444"),
        "Banana": ((234, 179, 8), "#EAB308"),
        "Orange": ((249, 115, 22), "#F97316"),
        "Strawberry": ((225, 29, 72), "#E11D48"),
        "Watermelon": ((16, 185, 129), "#10B981"),
        "Kiwi": ((132, 204, 22), "#84CC16"),
        "Lemon": ((250, 204, 21), "#FACC15"),
        "Blueberry": ((59, 130, 246), "#3B82F6"),
        "Grape": ((168, 85, 247), "#A855F7"),
        "Mango": ((251, 146, 60), "#FB923C"),
        "Pineapple": ((234, 179, 8), "#EAB308"),
        "Avocado": ((84, 110, 48), "#546E30"),
        "Cherry": ((190, 18, 60), "#BE123C"),
        "Peach": ((251, 146, 60), "#FB923C"),
        "Pear": ((163, 230, 53), "#A3E635"),
        "Pomegranate": ((159, 18, 57), "#9F1239"),
    }

    def __init__(self, classifier: Optional[FruitClassifier] = None):
        self.classifier = classifier or FruitClassifier()

    def _extract_proposals(
        self,
        image: Image.Image,
        min_area_ratio: float = 0.02
    ) -> List[Tuple[int, int, int, int]]:
        """
        Fast, scale-invariant region proposal extraction using OpenCV colorimetry,
        morphological filtering, and distance transform peak splitting for touching fruits.
        """
        orig_w, orig_h = image.size
        max_dim = 640
        scale = 1.0
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / float(max(orig_w, orig_h))
            sw, sh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            small = image.resize((sw, sh), Image.Resampling.BILINEAR)
        else:
            sw, sh = orig_w, orig_h
            small = image

        inv_scale = 1.0 / scale
        arr = np.array(small.convert("RGB"))

        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # 1. Border Sampling to estimate background color
        borders = np.concatenate([
            arr[:6, :, :].reshape(-1, 3),
            arr[-6:, :, :].reshape(-1, 3),
            arr[:, :6, :].reshape(-1, 3),
            arr[:, -6:, :].reshape(-1, 3)
        ], axis=0)
        bg_rgb = np.median(borders, axis=0)
        color_diff = np.linalg.norm(arr.astype(float) - bg_rgb, axis=2)

        # 2. Chromatic and Saliency features
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

        # 3. Morphological cleanup
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

            # Skip full-frame boundary contours (image frame artifacts)
            if (w_box >= 0.92 * sw and h_box >= 0.92 * sh) or area > (sw * sh * 0.85):
                continue

            # Primary contour bounding box with 4% padding
            pad_x = max(2, int(w_box * 0.04))
            pad_y = max(2, int(h_box * 0.04))
            bx1 = max(0, int((x - pad_x) * inv_scale))
            by1 = max(0, int((y - pad_y) * inv_scale))
            bx2 = min(orig_w, int((x + w_box + pad_x) * inv_scale))
            by2 = min(orig_h, int((y + h_box + pad_y) * inv_scale))
            boxes.append((bx1, by1, bx2, by2))

            # Distance transform peak splitting for touching/clustered fruits
            sub_mask = opened[y:y+h_box, x:x+w_box]
            dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
            max_dist = dist.max()

            if max_dist > 14.0 and (max(w_box, h_box) > 1.4 * min(w_box, h_box) or area > (sw * sh * 0.08)):
                dilated = cv2.dilate(dist, np.ones((13, 13), np.uint8))
                peaks = (dist == dilated) & (dist > 0.45 * max_dist)
                peak_indices = np.argwhere(peaks)
                if len(peak_indices) >= 2:
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

        # Fallback to full image if no distinct candidate was extracted
        if not boxes:
            boxes.append((0, 0, orig_w, orig_h))

        return boxes

    def detect(
        self,
        image: Image.Image,
        min_area_ratio: float = 0.02,
        max_boxes: int = 15
    ) -> MultiFruitDetectionResult:
        """
        Detects fruits using OpenCV morphological saliency, distance-transform peak separation,
        deep MobileNetV2 ONNX scoring, and universal geometric Non-Maximum Suppression.
        """
        w, h = image.size
        proposals = self._extract_proposals(image, min_area_ratio=min_area_ratio)

        # Classify all proposed candidate regions
        candidates = []
        for b in proposals:
            crop_w = b[2] - b[0]
            crop_h = b[3] - b[1]
            area = crop_w * crop_h
            if crop_w < 30 or crop_h < 30:
                continue

            crop = image.crop(b)
            pred = self.classifier.predict(crop)

            if pred.confidence >= 0.35 and pred.top_fruit not in ("Background", "Unknown / Background", "Unknown", "None"):
                candidates.append((b, pred.top_fruit, pred.confidence, area))

        # Universal Non-Maximum Suppression (IoU + Containment IoM)
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

        # Sort candidates by confidence * (area ** 0.25)
        candidates.sort(key=lambda c: c[2] * (c[3] ** 0.25), reverse=True)

        final_detections = []
        for box, fruit, score, area in candidates:
            suppress = False
            for fbox, ffruit, fscore, farea in final_detections:
                if iou(box, fbox) > 0.32 or iom(box, fbox) > 0.40:
                    suppress = True
                    break
            if not suppress:
                final_detections.append((box, fruit, score, area))

        # Fallback if no specific region proposal triggered
        if not final_detections:
            full_pred = self.classifier.predict(image)
            if full_pred.top_fruit not in ("Background", "Unknown / Background", "Unknown", "None"):
                final_detections.append(((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)), full_pred.top_fruit, full_pred.confidence, w * h))

        # Single fruit dominance check (avoids spurious fragment detection on clean single fruit images)
        if len(final_detections) > 1:
            primary = final_detections[0]
            prim_area = primary[3]
            if prim_area > (w * h * 0.40) and primary[2] > 0.85:
                all_fragments = all(iom(f[0], primary[0]) > 0.20 for f in final_detections[1:])
                if all_fragments:
                    final_detections = [primary]

        # Sort final detections left-to-right (by xmin) for orderly numbering
        final_detections.sort(key=lambda d: d[0][0])

        instances: List[DetectedFruitInstance] = []
        counts: Dict[str, int] = {}

        for idx, (box, fruit, score, _) in enumerate(final_detections[:max_boxes], start=1):
            norm_box = (
                round(box[0] / w, 3),
                round(box[1] / h, 3),
                round(box[2] / w, 3),
                round(box[3] / h, 3)
            )

            # Assign consistent fruit colors
            if fruit in self.FRUIT_COLOR_MAP:
                c_rgb, c_hex = self.FRUIT_COLOR_MAP[fruit]
            else:
                c_rgb = self.DEFAULT_PALETTE[(idx - 1) % len(self.DEFAULT_PALETTE)]
                c_hex = f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"

            inst = DetectedFruitInstance(
                instance_id=idx,
                fruit_name=fruit,
                confidence=score,
                box=box,
                normalized_box=norm_box,
                color_hex=c_hex,
                color_rgb=c_rgb
            )
            instances.append(inst)
            counts[fruit] = counts.get(fruit, 0) + 1

        annotated = self._render_annotations(image.copy(), instances)

        return MultiFruitDetectionResult(
            instances=instances,
            counts_by_fruit=counts,
            annotated_image=annotated,
            total_detected=len(instances)
        )

    def _render_annotations(
        self,
        image: Image.Image,
        instances: List[DetectedFruitInstance]
    ) -> Image.Image:
        """
        Draws crisp bounding boxes, color-coded badges, and clear ID tags.
        """
        draw = ImageDraw.Draw(image, "RGBA")
        w, h = image.size
        font = ImageFont.load_default()

        for inst in instances:
            color = inst.color_rgb
            x1, y1, x2, y2 = inst.box

            # Semi-transparent box fill (20% opacity) and crisp 3px border
            fill_color = (*color, 35)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=(*color, 240), width=3)

            # Badge label: "#1 Apple (93%)"
            label_text = f"#{inst.instance_id} {inst.fruit_name} ({inst.confidence * 100:.0f}%)"

            # Badge background
            badge_h = 22
            badge_w = len(label_text) * 7 + 14
            badge_y1 = max(0, y1 - badge_h)
            badge_y2 = badge_y1 + badge_h
            badge_x2 = min(w, x1 + badge_w)

            draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=(*color, 235))
            draw.text((x1 + 6, badge_y1 + 5), label_text, fill=(255, 255, 255), font=font)

        return image

    render_annotations = _render_annotations
