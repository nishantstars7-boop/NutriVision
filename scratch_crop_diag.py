import sys
sys.path.append('F:/Antig projects')
from PIL import Image
import numpy as np
from fruit_analyzer.models.classifier import FruitClassifier

clf = FruitClassifier()

# Let's test what MobileNetV2 ONNX session outputs by itself for various crops!
print("ONNX session exists:", clf.session is not None)

# Test an apple crop with some background
for name in ['apple.jpg', 'orange.jpg', 'banana.jpg', 'kiwi.jpg']:
    img = Image.open(f'fruit_analyzer/data/sample_images/{name}')
    w, h = img.size
    # tight crop (just the fruit)
    tight = img.crop((w//4, h//4, 3*w//4, 3*h//4))
    res = clf.predict(tight)
    print(f"Tight crop of {name}: {res.top_fruit} ({res.confidence:.2f}) -> {res.top_5}")
