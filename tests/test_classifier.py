from pathlib import Path
from PIL import Image
import pytest
from fruit_analyzer.models.classifier import FruitClassifier
from fruit_analyzer.models.detector import MultiFruitDetector


@pytest.fixture
def classifier():
    return FruitClassifier()


@pytest.fixture
def detector(classifier):
    return MultiFruitDetector(classifier)


def test_classifier_sample_apple(classifier):
    img_path = Path(__file__).resolve().parents[1] / "data" / "sample_images" / "apple.jpg"
    assert img_path.exists(), "Sample apple image missing."

    img = Image.open(img_path)
    res = classifier.predict(img)
    assert res.top_fruit == "Apple"
    assert res.confidence > 0.50
    assert len(res.top_5) > 0


def test_confidence_threshold_logic(classifier):
    img_path = Path(__file__).resolve().parents[1] / "data" / "sample_images" / "apple.jpg"
    img = Image.open(img_path)

    # Threshold set very high (0.99) -> should flag is_low_confidence = True
    res_high_thresh = classifier.predict(img, confidence_threshold=0.99)
    assert res_high_thresh.is_low_confidence is True

    # Threshold set very low (0.10) -> should flag is_low_confidence = False
    res_low_thresh = classifier.predict(img, confidence_threshold=0.10)
    assert res_low_thresh.is_low_confidence is False


def test_multi_fruit_detector(detector):
    img_path = Path(__file__).resolve().parents[1] / "data" / "sample_images" / "mixed_fruits.jpg"
    assert img_path.exists(), "Sample mixed fruit image missing."

    img = Image.open(img_path)
    res = detector.detect(img)
    assert res.total_detected >= 1
    assert len(res.instances) >= 1
    assert res.annotated_image is not None
