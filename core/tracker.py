"""
Daily Fruit & Nutrition Tracker
Provides logging, daily summation, progress tracking toward dietary targets,
and CSV export for consumed fruit portions.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict
import pandas as pd
from .quantity_calculator import CalculatedPortion


@dataclass
class LoggedFruitEntry:
    timestamp: str
    fruit_name: str
    quantity: float
    unit: str
    grams: float
    calories: float
    carbs_g: float
    protein_g: float
    fats_g: float
    fiber_g: float
    sugars_g: float
    glycemic_load: float


class DailyFruitTracker:
    """
    Session-based daily fruit intake logger.
    """

    WHO_RECOMMENDED_FRUIT_VEG_GRAMS = 400.0  # WHO 400g/day target

    def __init__(self):
        self.entries: List[LoggedFruitEntry] = []

    def log_portion(self, portion: CalculatedPortion):
        """Logs a calculated portion to today's log."""
        entry = LoggedFruitEntry(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fruit_name=portion.fruit_name,
            quantity=portion.input_quantity,
            unit=portion.unit.value,
            grams=portion.calculated_grams,
            calories=portion.calories,
            carbs_g=portion.total_carbs_g,
            protein_g=portion.protein_g,
            fats_g=portion.fats_g,
            fiber_g=portion.fiber_g,
            sugars_g=portion.sugars_g,
            glycemic_load=portion.glycemic_load
        )
        self.entries.append(entry)

    def get_summary(self) -> Dict[str, float]:
        """Calculates total consumed metrics for the active session."""
        if not self.entries:
            return {
                "total_items": 0,
                "total_grams": 0.0,
                "total_calories": 0.0,
                "total_carbs_g": 0.0,
                "total_protein_g": 0.0,
                "total_fats_g": 0.0,
                "total_fiber_g": 0.0,
                "total_sugars_g": 0.0,
                "who_progress_pct": 0.0
            }

        tot_grams = sum(e.grams for e in self.entries)
        who_pct = min(100.0, round((tot_grams / self.WHO_RECOMMENDED_FRUIT_VEG_GRAMS) * 100.0, 1))

        return {
            "total_items": len(self.entries),
            "total_grams": round(tot_grams, 1),
            "total_calories": round(sum(e.calories for e in self.entries), 1),
            "total_carbs_g": round(sum(e.carbs_g for e in self.entries), 1),
            "total_protein_g": round(sum(e.protein_g for e in self.entries), 1),
            "total_fats_g": round(sum(e.fats_g for e in self.entries), 1),
            "total_fiber_g": round(sum(e.fiber_g for e in self.entries), 1),
            "total_sugars_g": round(sum(e.sugars_g for e in self.entries), 1),
            "who_progress_pct": who_pct
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts entries to pandas DataFrame for display or download."""
        if not self.entries:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.entries])

    def clear(self):
        """Clears the current session log."""
        self.entries.clear()
