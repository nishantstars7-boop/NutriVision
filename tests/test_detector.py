import pytest
from PIL import Image
from pathlib import Path
from fruit_analyzer.models.detector import MultiFruitDetector

@pytest.fixture(scope="module")
def detector():
    return MultiFruitDetector()

@pytest.fixture(scope="module")
def samples_dir():
    return Path(__file__).resolve().parents[1] / "data" / "sample_images"

def test_single_fruit_detection_exact_count(detector, samples_dir):
    """Ensures single fruit photos never detect multiple items or fragments."""
    single_samples = ["apple.jpg", "banana.jpg", "orange.jpg", "strawberry.jpg", "watermelon.jpg", "kiwi.jpg"]
    for sample_name in single_samples:
        img_path = samples_dir / sample_name
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        result = detector.detect(img)
        assert result.total_detected == 1, f"Expected 1 fruit for {sample_name}, got {result.total_detected}: {[i.fruit_name for i in result.instances]}"

def test_multi_fruit_detection_count(detector, samples_dir):
    """Ensures multi-fruit images identify discrete multiple items."""
    multi_samples = ["two_bananas_one_apple.jpg", "mixed_fruits.jpg", "mixed_plate.jpg"]
    for sample_name in multi_samples:
        img_path = samples_dir / sample_name
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        result = detector.detect(img)
        assert result.total_detected >= 2, f"Expected at least 2 fruits for {sample_name}, got {result.total_detected}"

def test_instance_attribution_fields(detector, samples_dir):
    """Ensures every detected instance carries box id, color, confidence, and bounding box."""
    img_path = samples_dir / "two_bananas_one_apple.jpg"
    img = Image.open(img_path)
    result = detector.detect(img)
    for inst in result.instances:
        assert inst.instance_id >= 1
        assert len(inst.fruit_name) > 0
        assert inst.confidence >= 0.50
        assert inst.color_hex.startswith("#")
        assert len(inst.box) == 4
