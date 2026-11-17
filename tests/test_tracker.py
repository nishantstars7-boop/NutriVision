import pytest
from fruit_analyzer.core.nutrition_db import FruitNutrition
from fruit_analyzer.core.quantity_calculator import QuantityCalculator, PortionUnit
from fruit_analyzer.core.tracker import DailyFruitTracker


@pytest.fixture
def sample_portion():
    fruit = FruitNutrition(
        fruit_name="Orange",
        aliases=["navel orange"],
        category="Citrus",
        kcal_per_100g=47.0,
        avg_weight_per_piece_g=131.0,
        protein_g=0.9,
        carbs_g=11.8,
        fats_g=0.1,
        fiber_g=2.4,
        sugars_g=9.4,
        water_g=86.8,
        vitamin_c_mg=53.2,
        potassium_mg=181.0,
        glycemic_index=43,
        gi_category="Low",
        health_benefits="Rich in vitamin C."
    )
    return QuantityCalculator.calculate_portion(fruit, quantity=2.0, unit=PortionUnit.COUNT)


def test_tracker_logging_and_summary(sample_portion):
    tracker = DailyFruitTracker()
    assert tracker.get_summary()["total_items"] == 0

    tracker.log_portion(sample_portion)
    summary = tracker.get_summary()

    assert summary["total_items"] == 1
    # 2 oranges = 2 * 131 = 262g
    assert summary["total_grams"] == 262.0
    assert summary["total_calories"] > 0
    # WHO progress: (262 / 400) * 100 = 65.5%
    assert pytest.approx(summary["who_progress_pct"], 0.2) == 65.5


def test_tracker_clear(sample_portion):
    tracker = DailyFruitTracker()
    tracker.log_portion(sample_portion)
    assert len(tracker.entries) == 1

    tracker.clear()
    assert len(tracker.entries) == 0
    assert tracker.get_summary()["total_items"] == 0
