import sys
sys.path.append('F:/Antig projects')
from PIL import Image, ImageDraw
from fruit_analyzer.models.detector import MultiFruitDetector

det = MultiFruitDetector()

# A photo on a wooden table / cutting board (RGB 210, 180, 130)
# With an Apple and an Orange sitting on it
w, h = 600, 400
img = Image.new("RGB", (w, h), (210, 180, 130)) # Wood/cutting board background!
d = ImageDraw.Draw(img)
# Two fruits in the center
d.ellipse([150, 120, 290, 260], fill=(215, 30, 30)) # Apple
d.ellipse([310, 120, 450, 260], fill=(245, 130, 20)) # Orange

res = det.detect(img)
print("Scenario: Apple and Orange on wooden table / cutting board:")
for i in res.instances:
    print(f"  Box #{i.instance_id}: {i.fruit_name} ({i.confidence:.2f}) {i.box}")
