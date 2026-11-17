"""
Downloads real fruit images from the benchmark Fruits-360 dataset (Horea94/Fruit-Images-Dataset)
and prepares training/validation splits as well as real-world sample images for FruitVision.
"""

import os
import json
import urllib.request
import concurrent.futures
from pathlib import Path
from PIL import Image
import io

DATASET_ROOT = Path("fruit_analyzer/data/fruits_dataset")
SAMPLE_DIR = Path("fruit_analyzer/data/sample_images")

# Map of canonical fruit classes to Fruits-360 GitHub folder names
FRUIT_SOURCE_MAP = {
    "Apple": ["Apple Golden 1", "Apple Granny Smith", "Apple Red 1", "Apple Braeburn"],
    "Banana": ["Banana", "Banana Lady Finger"],
    "Orange": ["Orange", "Mandarine", "Clementine"],
    "Strawberry": ["Strawberry", "Strawberry Wedge"],
    "Watermelon": ["Watermelon"],
    "Kiwi": ["Kiwi"],
    "Lemon": ["Lemon", "Lemon Meyer"],
    "Mango": ["Mango", "Mango Red"],
    "Pineapple": ["Pineapple", "Pineapple Mini"],
    "Peach": ["Peach", "Peach Flat"],
    "Grape": ["Grape Blue", "Grape Pink", "Grape White"],
    "Avocado": ["Avocado", "Avocado ripe"],
    "Cherry": ["Cherry 1", "Cherry Rainier"],
    "Pomegranate": ["Pomegranate"],
    "Pear": ["Pear", "Pear Red"]
}

BASE_GITHUB_RAW = "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master"
GITHUB_API_BASE = "https://api.github.com/repos/Horea94/Fruit-Images-Dataset/contents"


def fetch_folder_contents(subpath: str):
    """Fetches list of filenames in a GitHub folder."""
    url = f"{GITHUB_API_BASE}/{subpath.replace(' ', '%20')}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return [item["name"] for item in data if item["type"] == "file" and item["name"].endswith(".jpg")]
    except Exception as e:
        print(f"Error listing {subpath}: {e}")
        return []


def download_single_image(url: str, save_path: Path):
    """Downloads and saves a single image safely."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists() and save_path.stat().st_size > 100:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
            # Verify valid image
            img = Image.open(io.BytesIO(content))
            img.verify()
            with open(save_path, "wb") as f:
                f.write(content)
            return True
    except Exception as e:
        return False


def download_dataset(train_per_class: int = 50, val_per_class: int = 20):
    """Downloads real training and validation images for all target fruit classes."""
    train_dir = DATASET_ROOT / "train"
    val_dir = DATASET_ROOT / "val"

    tasks = []
    print(f"Listing and preparing download for {len(FRUIT_SOURCE_MAP)} fruit classes...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for fruit_name, folders in FRUIT_SOURCE_MAP.items():
            print(f"Scanning remote repository for {fruit_name}...")
            # Query the first folder
            primary_folder = folders[0]
            files = fetch_folder_contents(f"Test/{primary_folder}")
            if not files:
                files = fetch_folder_contents(f"Training/{primary_folder}")

            # Also query secondary folder if needed
            if len(folders) > 1 and len(files) < (train_per_class + val_per_class):
                sec_files = fetch_folder_contents(f"Test/{folders[1]}")
                files.extend([f"{folders[1]}/{f}" for f in sec_files])

            # Select train and val files
            selected_train = files[:train_per_class]
            selected_val = files[train_per_class:train_per_class + val_per_class]

            for fname in selected_train:
                if "/" in fname:
                    sub_f, img_name = fname.split("/")
                    img_url = f"{BASE_GITHUB_RAW}/Test/{sub_f.replace(' ', '%20')}/{img_name}"
                else:
                    img_url = f"{BASE_GITHUB_RAW}/Test/{primary_folder.replace(' ', '%20')}/{fname}"
                dest = train_dir / fruit_name / f"{fruit_name}_{Path(fname).name}"
                tasks.append((img_url, dest))

            for fname in selected_val:
                if "/" in fname:
                    sub_f, img_name = fname.split("/")
                    img_url = f"{BASE_GITHUB_RAW}/Test/{sub_f.replace(' ', '%20')}/{img_name}"
                else:
                    img_url = f"{BASE_GITHUB_RAW}/Test/{primary_folder.replace(' ', '%20')}/{fname}"
                dest = val_dir / fruit_name / f"{fruit_name}_{Path(fname).name}"
                tasks.append((img_url, dest))

    print(f"Queued {len(tasks)} real fruit images to download. Downloading in parallel...")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_single_image, url, dest) for url, dest in tasks]
        for f in concurrent.futures.as_completed(futures):
            if f.result():
                success_count += 1

    print(f"Successfully downloaded {success_count} / {len(tasks)} real fruit images into {DATASET_ROOT}")


def download_real_sample_images():
    """Downloads real high-resolution fruit photos for the sample showcase."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    samples = {
        # Real high-resolution single fruits from test-multiple_fruits or open photo datasets
        "apple.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/apple.jpg",
        "banana.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/Bananas(lady_finger)1.jpg",
        "strawberry.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/strawberries.jpg",
        "kiwi.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/kiwi.jpg",
        
        # Real multi-fruit scenes
        "mixed_fruits.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/apple_grape.jpg",
        "mixed_plate.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/apple_apricot_peach_peach(flat)_pomegranate_pear_plum.jpg",
        "two_bananas_one_apple.jpg": "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/test-multiple_fruits/mangos_bananas(lady_finger).jpg",
    }

    print("Downloading real-world sample showcase images...")
    for target_name, url in samples.items():
        save_path = SAMPLE_DIR / target_name
        # Download and resize to a clean web-ready resolution (e.g. max 800px) so Streamlit renders instantly
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read()
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                img.save(save_path, "JPEG", quality=90)
                print(f"Saved real sample: {target_name} ({img.size})")
        except Exception as e:
            print(f"Failed to download {target_name}: {e}")

    # Also obtain real photos for Orange and Watermelon
    # Real Orange
    orange_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Orange-Fruit-Pieces.jpg/640px-Orange-Fruit-Pieces.jpg"
    try:
        req = urllib.request.Request(orange_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            img.save(SAMPLE_DIR / "orange.jpg", "JPEG", quality=90)
            print(f"Saved real sample: orange.jpg ({img.size})")
    except Exception as e:
        print(f"Error downloading orange: {e}")

    # Real Watermelon
    wm_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Watermelon.jpg/640px-Watermelon.jpg"
    try:
        req = urllib.request.Request(wm_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            img.save(SAMPLE_DIR / "watermelon.jpg", "JPEG", quality=90)
            print(f"Saved real sample: watermelon.jpg ({img.size})")
    except Exception as e:
        print(f"Error downloading watermelon: {e}")


if __name__ == "__main__":
    download_dataset(train_per_class=45, val_per_class=15)
    download_real_sample_images()
