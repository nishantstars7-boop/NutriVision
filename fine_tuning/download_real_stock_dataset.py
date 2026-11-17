"""
Download Real Stock Photography Dataset for FruitVision
Fast, multi-threaded collection of in-the-wild stock photographs for all 15 fruit classes
and natural background surfaces from Wikimedia Commons (using 800px pre-rendered thumbnails).
"""

import os
import io
import json
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

TARGET_DIR = Path("fruit_analyzer/data/real_stock_dataset")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FRUIT_QUERIES = {
    "Apple": [
        "fresh red apple fruit", "green granny smith apple", "apples in bowl basket", "red apple on wooden table",
        "red delicious apple fruit", "fuji apple fruit", "gala apple fruit", "apple harvest basket"
    ],
    "Banana": [
        "ripe yellow bananas bunch", "banana fruit on table", "fresh bananas in kitchen", "cavendish banana fruit",
        "bananas bunch on market", "peeled banana on plate", "fresh banana single"
    ],
    "Orange": [
        "fresh orange citrus fruit", "oranges in fruit bowl", "ripe orange sliced whole", "orange on wooden table",
        "valencia orange fruit", "navel orange citrus", "oranges on tree branch"
    ],
    "Strawberry": [
        "fresh ripe strawberries", "strawberries in bowl plate", "ripe strawberry fruit", "strawberries on cutting board",
        "strawberry basket fresh", "garden strawberry fruit", "strawberries macro"
    ],
    "Watermelon": [
        "watermelon slice fresh", "whole ripe watermelon", "cut watermelon fruit on plate", "fresh watermelon",
        "watermelon wedge ripe", "citrullus lanatus fruit", "sliced red watermelon"
    ],
    "Avocado": [
        "fresh ripe avocado sliced", "avocado fruit on cutting board", "whole hass avocado", "avocado on plate",
        "persea americana fruit", "half avocado seed", "avocado kitchen counter"
    ],
    "Kiwi": [
        "fresh kiwi fruit sliced", "kiwifruit on plate", "whole kiwi fruit", "kiwi on wooden table",
        "actinidia deliciosa fruit", "green kiwi sliced", "fuzzy kiwifruit"
    ],
    "Lemon": [
        "fresh yellow lemon fruit", "lemons on wooden table", "ripe lemon citrus sliced", "lemons in basket",
        "citrus limon fruit", "whole yellow lemon", "lemon slices on cutting board"
    ],
    "Mango": [
        "fresh ripe mango fruit", "mango sliced on plate", "mangoes fruit on table", "alphonso mango fruit",
        "mangifera indica fruit", "yellow ripe mango", "mango bowl kitchen"
    ],
    "Peach": [
        "fresh ripe peach fruit", "peaches in basket", "peach fruit on table", "fresh juicy peach",
        "prunus persica fruit", "yellow peach sliced", "peaches on wood"
    ],
    "Pear": [
        "fresh ripe pear fruit", "green yellow pear on table", "pears in bowl", "bosc pear fruit",
        "pyrus communis fruit", "bartlett pear fresh", "pear on cutting board"
    ],
    "Pineapple": [
        "fresh whole pineapple fruit", "pineapple fruit on table", "cut pineapple on plate", "ripe pineapple",
        "ananas comosus fruit", "pineapple crown fruit", "pineapple kitchen"
    ],
    "Pomegranate": [
        "fresh ripe pomegranate fruit", "pomegranate seeds open", "pomegranate fruit on plate", "whole pomegranate",
        "punica granatum fruit", "cut pomegranate fruit", "pomegranate on dark table"
    ],
    "Grape": [
        "bunch of fresh grapes", "red green grapes on plate", "grapes fruit cluster", "fresh table grapes",
        "vitis vinifera grapes", "purple grapes bunch", "white grapes cluster"
    ],
    "Cherry": [
        "fresh red cherries bunch", "cherries in bowl", "ripe sweet cherries on table", "cherry fruit fresh",
        "prunus avium fruit", "dark red cherries", "bing cherries on table"
    ],
    "Background": [
        "empty wooden cutting board", "empty ceramic plate bowl", "kitchen countertop marble slate",
        "empty wooden dining table", "textured kitchen background", "empty kitchen table surface",
        "clean kitchen counter top", "wooden table background texture", "white marble tabletop"
    ]
}

USER_AGENT = "FruitVisionResearchBot/2.0 (education_research@fruitvision.org)"


def search_wikimedia(query: str, limit: int = 25) -> list:
    encoded = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrnamespace=6&gsrsearch={encoded}"
        f"&gsrlimit={limit}&prop=imageinfo&iiprop=url&iiurlwidth=800"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            urls = []
            for pid, page in pages.items():
                title = page.get("title", "").lower()
                if any(ext in title for ext in [".svg", ".png", ".gif", "map", "flag", "icon", "logo", "diagram", "stamp"]):
                    continue
                ii = page.get("imageinfo", [{}])[0]
                thumb = ii.get("thumburl") or ii.get("url")
                if thumb:
                    clean = thumb.split("?")[0].lower()
                    if clean.endswith((".jpg", ".jpeg")):
                        urls.append(thumb)
            return urls
    except Exception as e:
        print(f"[Search error] {query}: {e}")
        return []


def download_and_save(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 3000:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
            if len(content) < 3000:
                return False
            img = Image.open(io.BytesIO(content))
            w, h = img.size
            if w < 100 or h < 100:
                return False
            # Normalize to max 800px
            if max(w, h) > 800:
                scale = 800.0 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            img.convert("RGB").save(dest_path, "JPEG", quality=90)
            return True
    except Exception:
        return False


def process_class(cls_name: str, queries: list, target_count: int = 75) -> int:
    cls_dir = TARGET_DIR / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(cls_dir.glob("*.jpg")))
    if existing >= target_count:
        print(f"[{cls_name}] Already complete ({existing} images).")
        return existing

    all_urls = []
    for q in queries:
        urls = search_wikimedia(q, limit=35)
        for u in urls:
            if u not in all_urls:
                all_urls.append(u)
        if len(all_urls) >= target_count * 2:
            break

    downloaded = existing
    for u in all_urls:
        if downloaded >= target_count:
            break
        dest = cls_dir / f"{cls_name.lower()}_{downloaded:03d}.jpg"
        if download_and_save(u, dest):
            downloaded += 1
            print(f"  + [{cls_name}] Saved {dest.name} ({downloaded}/{target_count})")
        time.sleep(0.05)

    print(f"[{cls_name}] Total: {downloaded} real stock photos.")
    return downloaded


def main():
    print("=" * 60)
    print("Downloading Real In-The-Wild Stock Photography Dataset")
    print("=" * 60)

    total = 0
    TARGET_PER_CLASS = 75  # 75 real stock photos per class x 16 classes = 1,200 real photos

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_class, cls_name, queries, TARGET_PER_CLASS): cls_name
            for cls_name, queries in FRUIT_QUERIES.items()
        }
        for future in as_completed(futures):
            cls_name = futures[future]
            try:
                cnt = future.result()
                total += cnt
            except Exception as e:
                print(f"[{cls_name}] Error: {e}")

    print("=" * 60)
    print(f"Real Stock Photography Download Complete! Total images: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
