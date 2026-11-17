import pytest
from fruit_analyzer.core.nutrition_db import FruitNutrition
from fruit_analyzer.core.quantity_calculator import QuantityCalculator, PortionUnit


@pytest.fixture
def sample_apple():
    return FruitNutrition(
        fruit_name="Apple",
        aliases=["red apple"],
        category="Pome",
        kcal_per_100g=52.0,
        avg_weight_per_piece_g=182.0,
        protein_g=0.3,
        carbs_g=13.8,
        fats_g=0.2,
        fiber_g=2.4,
        sugars_g=10.4,
        water_g=85.6,
        vitamin_c_mg=4.6,
        potassium_mg=107.0,
        glycemic_index=36,
        gi_category="Low",
        health_benefits="Pectin fiber supports gut microbiome."
    )


def test_quantity_count_formula(sample_apple):
    # 2 apples = 2 * 182 = 364g
    # calories = (364 / 100) * 52 = 189.28 -> 189.3 kcal
    portion = QuantityCalculator.calculate_portion(sample_apple, quantity=2.0, unit=PortionUnit.COUNT)
    assert portion.calculated_grams == 364.0
    assert pytest.approx(portion.calories, 0.2) == 189.3
    # Net carbs = 3.64 * (13.8 - 2.4) = 41.5g
    assert pytest.approx(portion.net_carbs_g, 0.2) == 41.5


def test_quantity_grams_formula(sample_apple):
    # 100g exact input
    portion = QuantityCalculator.calculate_portion(sample_apple, quantity=100.0, unit=PortionUnit.GRAMS)
    assert portion.calculated_grams == 100.0
    assert portion.calories == 52.0
    assert portion.fiber_g == 2.4
    assert portion.total_carbs_g == 13.8


def test_ounces_conversion(sample_apple):
    # 1 oz = 28.35g -> calories = (28.35 / 100) * 52 = 14.7 kcal
    portion = QuantityCalculator.calculate_portion(sample_apple, quantity=1.0, unit=PortionUnit.OUNCES)
    assert pytest.approx(portion.calculated_grams, 0.1) == 28.3
    assert pytest.approx(portion.calories, 0.2) == 14.7


def test_glycemic_load(sample_apple):
    # For 1 apple (182g):
    # net_carbs = 1.82 * 11.4 = 20.75g
    # GL = (36 * 20.75) / 100 = 7.47 -> Low category
    portion = QuantityCalculator.calculate_portion(sample_apple, quantity=1.0, unit=PortionUnit.COUNT)
    assert portion.glycemic_load <= 10.0
    assert portion.gl_category == "Low"


def test_multi_item_aggregation(sample_apple):
    p1 = QuantityCalculator.calculate_portion(sample_apple, quantity=1.0, unit=PortionUnit.COUNT)
    p2 = QuantityCalculator.calculate_portion(sample_apple, quantity=1.0, unit=PortionUnit.COUNT)

    agg = QuantityCalculator.aggregate_items([p1, p2])
    assert agg.total_grams == pytest.approx(364.0, 0.2)
    assert agg.total_calories == pytest.approx(189.3, 0.3)
    assert "Apple" in agg.calorie_breakdown_by_fruit
