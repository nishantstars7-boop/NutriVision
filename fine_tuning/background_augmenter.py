"""
Realistic Background Synthesizer & Dataset Augmenter for Fruits
Blends isolated fruit images onto realistic backgrounds (wood, marble, slate, cloth, natural table tones)
with lighting gradients and shadows to eliminate background bias.
"""

import numpy as np
from typing import Optional
from PIL import Image, ImageFilter, ImageDraw
import random

def generate_realistic_background(size=(224, 224)) -> Image.Image:
    """
    Generates a realistic background simulating dining tables, cutting boards,
    marble counters, kitchen slates, or plates.
    """
    w, h = size
    bg_type = random.choice(["wood", "marble", "dark_slate", "warm_table", "cloth", "plate"])
    
    if bg_type == "wood":
        # Warm wood tone: brown, tan, amber
        base_color = np.array([
            random.randint(140, 210),  # R
            random.randint(90, 150),   # G
            random.randint(50, 100)    # B
        ], dtype=np.float32)
        # Add wood grain lines
        noise = np.random.normal(0, 12, (h, w, 1))
        # Horizontal/vertical grain stripes
        x = np.linspace(0, random.uniform(5, 15), w)
        grain = (np.sin(x) * 15)[:, np.newaxis]
        arr = base_color + noise + grain
        
    elif bg_type == "marble":
        # White/grey marble
        base_val = random.randint(210, 240)
        base_color = np.array([base_val, base_val, base_val], dtype=np.float32)
        noise = np.random.normal(0, 8, (h, w, 3))
        # Soft marble veins
        arr = base_color + noise
        
    elif bg_type == "dark_slate":
        # Dark slate / cutting board
        val = random.randint(40, 80)
        base_color = np.array([val, val, val + random.randint(0, 10)], dtype=np.float32)
        noise = np.random.normal(0, 6, (h, w, 3))
        arr = base_color + noise
        
    elif bg_type == "warm_table":
        # Warm beige/cream tabletop
        base_color = np.array([
            random.randint(220, 245),
            random.randint(210, 235),
            random.randint(190, 215)
        ], dtype=np.float32)
        noise = np.random.normal(0, 5, (h, w, 3))
        arr = base_color + noise
        
    elif bg_type == "cloth":
        # Fabric / napkin pattern
        r_tone = random.randint(180, 230)
        g_tone = random.randint(180, 230)
        b_tone = random.randint(180, 230)
        base_color = np.array([r_tone, g_tone, b_tone], dtype=np.float32)
        # Weave pattern
        x = np.arange(w) % 4 < 2
        y = np.arange(h) % 4 < 2
        weave = ((x[:, None] ^ y[None, :]) * 10).astype(np.float32)[:, :, np.newaxis]
        arr = base_color + weave
        
    else:  # plate
        # Circular plate highlight
        arr = np.ones((h, w, 3), dtype=np.float32) * random.randint(220, 250)
        yy, xx = np.mgrid[:h, :w]
        dist = np.sqrt((xx - w/2)**2 + (yy - h/2)**2)
        plate_rim = (np.abs(dist - w*0.42) < 6) * -30
        arr += plate_rim[:, :, np.newaxis]

    # Add lighting gradient / ambient shadow
    grad_x = np.linspace(-15, 15, w)[np.newaxis, :, np.newaxis]
    grad_y = np.linspace(-15, 15, h)[:, np.newaxis, np.newaxis]
    arr = arr + grad_x + grad_y
    
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def blend_fruit_on_background(fruit_img: Image.Image, bg_img: Optional[Image.Image] = None) -> Image.Image:
    """
    Extracts fruit foreground from pure white Fruits-360 background and blends it
    seamlessly onto a realistic background with soft contact shadow.
    """
    fruit_rgb = fruit_img.convert("RGB")
    w, h = fruit_rgb.size
    
    # Fruits-360 background is pure #FFFFFF (or > 250 across all channels)
    arr = np.array(fruit_rgb, dtype=np.float32)
    # Mask of fruit pixels: where not near-white
    is_fg = (arr[:, :, 0] < 248) | (arr[:, :, 1] < 248) | (arr[:, :, 2] < 248)
    
    # If no background provided, generate one
    if bg_img is None:
        bg_img = generate_realistic_background((w, h))
    else:
        bg_img = bg_img.resize((w, h))
        
    # Create soft alpha mask with feathering for natural edges
    mask = Image.fromarray((is_fg * 255).astype(np.uint8), mode="L")
    mask_blurred = mask.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    # Contact shadow under fruit
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    # Get bounding box of foreground
    bbox = mask.getbbox()
    if bbox:
        x0, y0, x1, y1 = bbox
        shadow_box = [x0 + 4, y1 - 6, x1 - 4, min(h, y1 + 10)]
        draw.ellipse(shadow_box, fill=(20, 20, 20, 90))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=4))
        
    result = bg_img.convert("RGBA")
    result.alpha_composite(shadow)
    result.paste(fruit_rgb, (0, 0), mask_blurred)
    return result.convert("RGB")


class BackgroundAugmenter:
    """Wrapper class providing an augment() method for fruit images."""
    def augment(self, fruit_img: Image.Image) -> Image.Image:
        return blend_fruit_on_background(fruit_img)

