"""
Fruit Calorie & Nutrition Database Engine
Provides authoritative nutritional data from local curated CSV with RapidFuzz
fuzzy matching, alias resolution, and optional live USDA FoodData Central API fetching.
"""

import os
import csv
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from pathlib import Path

try:
    from rapidfuzz import process, fuzz
except ImportError:
    # Graceful fallback if rapidfuzz is not yet on path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages"))
    from rapidfuzz import process, fuzz


@dataclass
class FruitNutrition:
    fruit_name: str
    aliases: List[str]
    category: str
    kcal_per_100g: float
    avg_weight_per_piece_g: float
    protein_g: float
    carbs_g: float
    fats_g: float
    fiber_g: float
    sugars_g: float
    water_g: float
    vitamin_c_mg: float
    potassium_mg: float
    glycemic_index: int
    gi_category: str
    health_benefits: str
    usda_fdc_id: Optional[int] = None
    data_source: str = "Local Curated DB (USDA SR Legacy Aligned)"


class NutritionDatabase:
    """
    Manages fruit nutritional reference data with fuzzy search and optional USDA FoodData Central API sync.
    """

    def __init__(self, csv_path: Optional[str] = None, cache_db_path: Optional[str] = None):
        if csv_path is None:
            csv_path = str(Path(__file__).resolve().parents[1] / "data" / "fruit_nutrition.csv")
        self.csv_path = Path(csv_path)

        if cache_db_path is None:
            cache_db_path = str(Path(__file__).resolve().parents[1] / "data" / "usda_cache.sqlite")
        self.cache_db_path = Path(cache_db_path)

        self._fruits: Dict[str, FruitNutrition] = {}
        self._lookup_keys: Dict[str, str] = {}  # alias/key -> canonical fruit_name
        self._load_csv()
        self._init_sqlite_cache()

    def _init_sqlite_cache(self):
        """Initializes SQLite cache for runtime USDA API queries."""
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usda_cache (
                    query_name TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_csv(self):
        """Parses the curated CSV into structured dataclasses."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Nutrition database CSV not found at: {self.csv_path}")

        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["fruit_name"].strip()
                aliases = [a.strip().lower() for a in row.get("aliases", "").split(",") if a.strip()]
                fdc_id = int(row["usda_fdc_id"]) if row.get("usda_fdc_id") and row["usda_fdc_id"].isdigit() else None

                fruit = FruitNutrition(
                    fruit_name=name,
                    aliases=aliases,
                    category=row.get("category", "Fruit").strip(),
                    kcal_per_100g=float(row.get("kcal_per_100g", 0)),
                    avg_weight_per_piece_g=float(row.get("avg_weight_per_piece_g", 100)),
                    protein_g=float(row.get("protein_g", 0)),
                    carbs_g=float(row.get("carbs_g", 0)),
                    fats_g=float(row.get("fats_g", 0)),
                    fiber_g=float(row.get("fiber_g", 0)),
                    sugars_g=float(row.get("sugars_g", 0)),
                    water_g=float(row.get("water_g", 0)),
                    vitamin_c_mg=float(row.get("vitamin_c_mg", 0)),
                    potassium_mg=float(row.get("potassium_mg", 0)),
                    glycemic_index=int(row.get("glycemic_index", 40)),
                    gi_category=row.get("gi_category", "Low").strip(),
                    health_benefits=row.get("health_benefits", "").strip(),
                    usda_fdc_id=fdc_id
                )

                self._fruits[name.lower()] = fruit
                self._lookup_keys[name.lower()] = name.lower()
                for alias in aliases:
                    self._lookup_keys[alias] = name.lower()

    def get_fruit(self, query: str, score_cutoff: float = 65.0) -> Optional[FruitNutrition]:
        """
        Retrieves fruit nutrition by exact or fuzzy match across canonical names and known aliases.
        """
        if not query:
            return None

        clean_query = query.strip().lower()

        # 1. Exact match in canonical names or aliases
        if clean_query in self._lookup_keys:
            canonical = self._lookup_keys[clean_query]
            return self._fruits[canonical]

        # 2. Fuzzy search across all known names and aliases
        candidates = list(self._lookup_keys.keys())
        match = process.extractOne(
            clean_query,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff
        )

        if match:
            matched_key, score, _ = match
            canonical = self._lookup_keys[matched_key]
            return self._fruits[canonical]

        # 3. Check SQLite cache
        cached = self._get_cached_usda(clean_query)
        if cached:
            return cached

        return None

    def fetch_from_usda_api(self, fruit_query: str, api_key: str = "DEMO_KEY") -> Optional[FruitNutrition]:
        """
        Fetches official nutrient breakdown from USDA FoodData Central API (v1) and caches locally.
        Gracefully handles network offline or missing key.
        """
        clean_query = fruit_query.strip().lower()

        # Check cache first
        cached = self._get_cached_usda(clean_query)
        if cached:
            return cached

        url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": f"{fruit_query} raw",
            "dataType": ["Foundation", "SR Legacy"],
            "pageSize": 1,
            "api_key": api_key
        }

        try:
            import httpx
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(url, params=params)
                if resp.status_code != 200:
                    return None
                data = resp.json()

            foods = data.get("foods", [])
            if not foods:
                return None

            top_food = foods[0]
            fdc_id = top_food.get("fdcId")
            description = top_food.get("description", fruit_query)

            nutrients = {n.get("nutrientName", "").lower(): n.get("value", 0.0) for n in top_food.get("foodNutrients", [])}

            kcal = nutrients.get("energy", 50.0)
            protein = nutrients.get("protein", 0.5)
            carbs = nutrients.get("carbohydrate, by difference", 12.0)
            fats = nutrients.get("total lipid (fat)", 0.2)
            fiber = nutrients.get("fiber, total dietary", 2.0)
            sugars = nutrients.get("sugars, total including nlea", 9.0)
            vit_c = nutrients.get("vitamin c, total ascorbic acid", 10.0)
            potassium = nutrients.get("potassium, k", 150.0)
            water = nutrients.get("water", 85.0)

            fruit = FruitNutrition(
                fruit_name=fruit_query.capitalize(),
                aliases=[description.lower()],
                category="USDA Live",
                kcal_per_100g=float(kcal),
                avg_weight_per_piece_g=150.0,
                protein_g=float(protein),
                carbs_g=float(carbs),
                fats_g=float(fats),
                fiber_g=float(fiber),
                sugars_g=float(sugars),
                water_g=float(water),
                vitamin_c_mg=float(vit_c),
                potassium_mg=float(potassium),
                glycemic_index=45,
                gi_category="Low",
                health_benefits="Fetched directly from USDA FoodData Central database.",
                usda_fdc_id=fdc_id,
                data_source=f"USDA FoodData Central (FDC ID: {fdc_id})"
            )

            self._save_cached_usda(clean_query, fruit)
            return fruit

        except Exception:
            # Network offline or API issue - fallback safely
            return None

    def _get_cached_usda(self, query: str) -> Optional[FruitNutrition]:
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data_json FROM usda_cache WHERE query_name = ?", (query,))
                row = cursor.fetchone()
                if row:
                    d = json.loads(row[0])
                    return FruitNutrition(**d)
        except Exception:
            pass
        return None

    def _save_cached_usda(self, query: str, fruit: FruitNutrition):
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO usda_cache (query_name, data_json) VALUES (?, ?)",
                    (query, json.dumps(asdict(fruit)))
                )
        except Exception:
            pass

    def list_all_fruits(self) -> List[str]:
        """Returns sorted list of all unique canonical fruit names."""
        return sorted([f.fruit_name for f in self._fruits.values()])

    def get_all_records(self) -> Dict[str, FruitNutrition]:
        """Returns map of canonical names to nutrition records."""
        return dict(self._fruits)
