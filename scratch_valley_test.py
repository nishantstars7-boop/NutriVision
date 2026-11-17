import sys
sys.path.append('F:/Antig projects')
from PIL import Image, ImageDraw
import numpy as np

# Test valley splitting on 2 touching apples of same color
w, h = 500, 300
img = Image.new("RGB", (w, h), (245, 245, 245))
d = ImageDraw.Draw(img)
d.ellipse([100, 70, 240, 210], fill=(210, 35, 30))
d.ellipse([220, 70, 360, 210], fill=(210, 35, 30))

arr = np.array(img, dtype=np.float32)
r, g, b = arr[:, :, 0]/255., arr[:, :, 1]/255., arr[:, :, 2]/255.
diff = np.sqrt(np.sum((arr - 245) ** 2, axis=2))
mask = diff > 20.0

# Horizontal projection across x
h_proj = np.sum(mask, axis=0) # shape (w,)
# Valley detection between min_x and max_x
xs = np.where(h_proj > 0)[0]
min_x, max_x = xs[0], xs[-1]
sub_proj = h_proj[min_x:max_x]
mid = len(sub_proj) // 2
min_val = np.min(sub_proj[len(sub_proj)//4 : 3*len(sub_proj)//4])
valley_x = min_x + len(sub_proj)//4 + np.argmin(sub_proj[len(sub_proj)//4 : 3*len(sub_proj)//4])
print(f"Object from x={min_x} to {max_x}, detected valley neck at x={valley_x} (neck height={min_val} vs peak={np.max(sub_proj)})")
