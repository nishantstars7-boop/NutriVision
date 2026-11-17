import sys
sys.path.append('F:/Antig projects')
from PIL import Image, ImageDraw
import numpy as np
from fruit_analyzer.models.detector import MultiFruitDetector

det = MultiFruitDetector()

# Test Scenario 1: A photo with two apples
# Create a synthetic image of 2 red apples on a wooden/neutral table (rgb 220, 200, 180)
w, h = 600, 400
img1 = Image.new("RGB", (w, h), (220, 205, 185))
d1 = ImageDraw.Draw(img1)
# Apple 1
d1.ellipse([100, 120, 260, 280], fill=(210, 30, 25))
# Apple 2
d1.ellipse([340, 130, 500, 290], fill=(205, 35, 30))

res1 = det.detect(img1)
print("Scenario 1 (Two apples on table):", [(i.instance_id, i.fruit_name, f"{i.confidence:.2f}") for i in res1.instances])

# Test Scenario 2: Two oranges on a table
img2 = Image.new("RGB", (w, h), (230, 220, 210))
d2 = ImageDraw.Draw(img2)
d2.ellipse([100, 120, 260, 280], fill=(245, 130, 20))
d2.ellipse([340, 130, 500, 290], fill=(240, 125, 25))
res2 = det.detect(img2)
print("Scenario 2 (Two oranges on table):", [(i.instance_id, i.fruit_name, f"{i.confidence:.2f}") for i in res2.instances])

# Test Scenario 3: Apple + Orange touching on plate
img3 = Image.new("RGB", (w, h), (240, 240, 240))
d3 = ImageDraw.Draw(img3)
# Plate
d3.ellipse([100, 60, 500, 340], fill=(255, 255, 255), outline=(210, 210, 210), width=4)
# Apple
d3.ellipse([160, 110, 310, 260], fill=(220, 25, 25))
# Orange touching
d3.ellipse([290, 120, 440, 270], fill=(245, 130, 20))
res3 = det.detect(img3)
print("Scenario 3 (Apple + Orange touching on plate):", [(i.instance_id, i.fruit_name, f"{i.confidence:.2f}") for i in res3.instances])

# Test Scenario 4: What if an image has warm yellowish indoor lighting or low saturation background?
img4 = Image.new("RGB", (w, h), (210, 190, 150)) # warm table
d4 = ImageDraw.Draw(img4)
d4.ellipse([150, 120, 280, 250], fill=(200, 50, 40))
d4.ellipse([320, 110, 460, 260], fill=(70, 140, 50))
res4 = det.detect(img4)
print("Scenario 4 (Apple + Kiwi on warm wooden table):", [(i.instance_id, i.fruit_name, f"{i.confidence:.2f}") for i in res4.instances])
