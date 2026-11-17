"""
Download Expanded Real Fruit Dataset
Downloads diverse fruit varieties from Fruits-360 (green/red/golden apples, lady finger bananas,
meyer lemons, rainier cherries, white/blue grapes, flat peaches, etc.) and real multi-fruit test scenes.
"""

import os
import io
import json
import urllib.request
import concurrent.futures
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path("fruit_analyzer/data/fruits_dataset_expanded")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
MULTI_DIR = DATASET_ROOT / "multi_fruit_scenes"

BASE_RAW_URL = "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master"
GITHUB_API_DIR = "https://api.github.com/repos/Horea94/Fruit-Images-Dataset/contents"

# Mapping of our canonical 15 fruit classes to diverse Fruits-360 sub-folders
CLASS_SUBFOLDERS = {
    "Apple": [
        "Apple Braeburn", "Apple Golden 1", "Apple Granny Smith",
        "Apple Red 1", "Apple Red Delicious", "Apple Pink Lady", "Apple Crimson Snow"
    ],
    "Banana": [
        "Banana", "Banana Lady Finger", "Banana Red"
    ],
    "Orange": [
        "Orange"
    ],
    "Strawberry": [
        "Strawberry", "Strawberry Wedge"
    ],
    "Watermelon": [
        "Watermelon"
    ],
    "Avocado": [
        "Avocado", "Avocado ripe"
    ],
    "Kiwi": [
        "Kiwi"
    ],
    "Lemon": [
        "Lemon", "Lemon Meyer"
    ],
    "Mango": [
        "Mango", "Mango Red"
    ],
    "Peach": [
        "Peach", "Peach 2", "Peach Flat"
    ],
    "Pear": [
        "Pear", "Pear Red", "Pear Forelle", "Pear Abate", "Pear Williams"
    ],
    "Pineapple": [
        "Pineapple", "Pineapple Mini"
    ],
    "Pomegranate": [
        "Pomegranate"
    ],
    "Grape": [
        "Grape Blue", "Grape Pink", "Grape White"
    ],
    "Cherry": [
        "Cherry 1", "Cherry Rainier", "Cherry Wax Red"
    ],
}


def download_single_image(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 500:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as response:
            data = response.read()
            if len(data) > 500:
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
    except Exception:
        pass
    return False


def get_github_folder_files(folder_name: str) -> list:
    encoded_folder = urllib.parse.quote(folder_name)
    url = f"{GITHUB_API_DIR}/Training/{encoded_folder}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [item["name"] for item in data if item["name"].endswith(".jpg")]
    except Exception as e:
        print(f"Error fetching directory list for {folder_name}: {e}")
        return []


def main():
    print("=" * 70)
    print("Downloading Expanded Real Fruit Dataset (Diverse Varieties & Colors)")
    print("=" * 70)

    download_tasks = []

    for canonical_class, subfolders in CLASS_SUBFOLDERS.items():
        print(f"Collecting file list for {canonical_class} (from {len(subfolders)} varieties)...")
        train_target = TRAIN_DIR / canonical_class
        val_target = VAL_DIR / canonical_class
        train_target.mkdir(parents=True, exist_ok=True)
        val_target.mkdir(parents=True, exist_ok=True)

        images_per_subfolder = max(25, 160 // len(subfolders))

        for subfolder in subfolders:
            filenames = get_github_folder_files(subfolder)
            if not filenames:
                continue

            # Split between train and val
            n_select = min(len(filenames), images_per_subfolder)
            selected = filenames[:n_select]
            n_val = max(3, int(n_select * 0.20))
            val_files = selected[:n_val]
            train_files = selected[n_val:]

            encoded_folder = urllib.parse.quote(subfolder)
            for fname in train_files:
                url = f"{BASE_RAW_URL}/Training/{encoded_folder}/{urllib.parse.quote(fname)}"
                out_name = f"{subfolder.replace(' ', '_')}_{fname}"
                download_tasks.append((url, train_target / out_name))

            for fname in val_files:
                url = f"{BASE_RAW_URL}/Training/{encoded_folder}/{urllib.parse.quote(fname)}"
                out_name = f"{subfolder.replace(' ', '_')}_{fname}"
                download_tasks.append((url, val_target / out_name))

    # Also download multi-fruit test scenes
    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(f"{GITHUB_API_DIR}/test-multiple_fruits", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            scenes = json.loads(resp.read().decode("utf-8"))
            for s in scenes:
                if s["name"].endswith(".jpg"):
                    sname = s["name"]
                    url = f"{BASE_RAW_URL}/test-multiple_fruits/{urllib.parse.quote(sname)}"
                    download_tasks.append((url, MULTI_DIR / sname))
    except Exception as e:
        print(f"Warning fetching multi-fruit scenes: {e}")

    print(f"\nTotal image download tasks queued: {len(download_tasks)}")
    print("Downloading in parallel with 16 workers...")

    done_cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(download_single_image, url, path) for url, path in download_tasks]
        for f in concurrent.futures.as_completed(futures):
            if f.result():
                done_cnt += 1

    print(f"\nDownload completed: {done_cnt} / {len(download_tasks)} images downloaded successfully!")

if __name__ == "__main__":
    import urllib.parse
    main()
