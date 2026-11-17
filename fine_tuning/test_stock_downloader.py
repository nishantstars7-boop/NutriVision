import os
import json
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

TARGET_DIR = Path("fruit_analyzer/data/real_stock_photos")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

FRUIT_QUERIES = {
    "Apple": ["fresh red apple fruit", "green apple fruit on table", "apples in bowl basket"],
    "Banana": ["ripe yellow bananas bunch", "banana fruit on table", "fresh banana"],
    "Orange": ["fresh orange citrus fruit", "oranges in fruit bowl", "ripe orange sliced whole"],
    "Strawberry": ["fresh ripe strawberries", "strawberries in bowl plate", "strawberry fruit"],
    "Watermelon": ["watermelon slice fresh", "whole ripe watermelon", "cut watermelon fruit"],
    "Avocado": ["fresh ripe avocado sliced", "avocado fruit on cutting board", "whole avocado"],
    "Kiwi": ["fresh kiwi fruit sliced", "kiwifruit on plate", "whole kiwi fruit"],
    "Lemon": ["fresh yellow lemon fruit", "lemons on wooden table", "ripe lemon citrus"],
    "Mango": ["fresh ripe mango fruit", "mango sliced on plate", "mangoes fruit"],
    "Peach": ["fresh ripe peach fruit", "peaches in basket", "peach fruit on table"],
    "Pear": ["fresh ripe pear fruit", "green yellow pear on table", "pears in bowl"],
    "Pineapple": ["fresh whole pineapple fruit", "pineapple fruit on table", "cut pineapple"],
    "Pomegranate": ["fresh ripe pomegranate fruit", "pomegranate seeds open", "pomegranate fruit"],
    "Grape": ["bunch of fresh grapes", "red green grapes on plate", "grapes fruit cluster"],
    "Cherry": ["fresh red cherries bunch", "cherries in bowl", "ripe sweet cherries"],
    "Background": ["wooden cutting board empty", "kitchen countertop marble slate", "empty dining table wood", "empty ceramic plate bowl", "neutral wall background"]
}

def search_wikimedia(query: str, limit: int = 15) -> list:
    encoded = urllib.parse.quote(query)
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search&gsrnamespace=6&gsrsearch=filetype:bitmap%20{encoded}&gsrlimit={limit}&prop=imageinfo&iiprop=url"
    req = urllib.request.Request(url, headers={"User-Agent": "FruitVisionResearcher/2.0 (research@example.edu)"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            urls = []
            for pid, page in pages.items():
                ii = page.get("imageinfo", [{}])[0]
                img_url = ii.get("url")
                if img_url and (img_url.lower().endswith(".jpg") or img_url.lower().endswith(".jpeg")):
                    urls.append(img_url)
            return urls
    except Exception as e:
        print(f"Search failed for '{query}': {e}")
        return []

def download_and_validate(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "FruitVisionResearcher/2.0 (research@example.edu)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            if len(content) < 3000:
                return False
            # Validate with PIL
            img = Image.open(io.BytesIO(content))
            img.verify()
            with open(dest_path, "wb") as f:
                f.write(content)
            return True
    except Exception:
        return False

if __name__ == "__main__":
    import io
    print("Testing real stock photo download for Apple, Banana, Orange, Background...")
    for fruit in ["Apple", "Banana", "Orange", "Background"]:
        fruit_dir = TARGET_DIR / fruit
        fruit_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for q in FRUIT_QUERIES[fruit]:
            urls = search_wikimedia(q, limit=5)
            for u in urls:
                fname = f"{fruit}_{count:03d}.jpg"
                if download_and_validate(u, fruit_dir / fname):
                    count += 1
                if count >= 6:
                    break
            if count >= 6:
                break
        print(f"Downloaded {count} real stock photos for {fruit}.")
