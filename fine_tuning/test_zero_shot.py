import time
from PIL import Image
from transformers import pipeline

print('Loading zero-shot detector...')
t0 = time.time()
try:
    detector = pipeline('zero-shot-object-detection', model='google/owlv2-base-patch16-ensemble', device='cpu')
    print(f'Model loaded in {time.time()-t0:.2f}s')

    img = Image.open('fruit_analyzer/data/sample_images/two_bananas_one_apple.jpg')
    queries = ['banana', 'apple', 'orange', 'fruit']
    t1 = time.time()
    preds = detector(img, candidate_labels=queries, threshold=0.15)
    print(f'Inference in {time.time()-t1:.2f}s, Detections: {len(preds)}')
    for p in preds:
        print(p['label'], f"{p['score']*100:.1f}%", p['box'])
except Exception as e:
    print('Error loading or running OWLv2:', e)
