"""
Quantity Logic & Advanced Nutritional Calculation Engine
Computes exact calories, macronutrients, micronutrients, glycemic load,
and multi-fruit aggregates based on user-specified portions (count, grams, ounces, cups).
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from .nutrition_db import FruitNutrition


class PortionUnit(str, Enum):
    COUNT = "count"       # number of pieces
    GRAMS = "grams"       # direct weight in grams
    OUNCES = "ounces"     # direct weight in ounces
    CUPS = "cups"         # volumetric cups (~150g avg sliced fruit)


@dataclass
class CalculatedPortion:
    fruit_name: str
    input_quantity: float
    unit: PortionUnit
    calculated_grams: float
    calories: float
    protein_g: float
    total_carbs_g: float
    fiber_g: float
    net_carbs_g: float
    sugars_g: float
    fats_g: float
    water_g: float
    vitamin_c_mg: float
    vitamin_c_dv_pct: float
    potassium_mg: float
    potassium_dv_pct: float
    glycemic_index: int
    glycemic_load: float
    gl_category: str
    health_benefits: str
    data_source: str


@dataclass
class AggregateNutritionReport:
    items: List[CalculatedPortion]
    total_grams: float
    total_calories: float
    total_protein_g: float
    total_carbs_g: float
    total_fiber_g: float
    total_net_carbs_g: float
    total_sugars_g: float
    total_fats_g: float
    total_vitamin_c_mg: float
    total_potassium_mg: float
    calorie_breakdown_by_fruit: Dict[str, float]
    macro_distribution_pct: Dict[str, float]  # carbs, protein, fat


class QuantityCalculator:
    """
    Handles portion conversions, mathematical formulas, and nutritional aggregations.
    """

    GRAMS_PER_OUNCE = 28.3495231
    GRAMS_PER_CUP_SLICED = 150.0
    VITAMIN_C_DAILY_VALUE_MG = 90.0   # FDA Daily Value
    POTASSIUM_DAILY_VALUE_MG = 4700.0 # FDA Daily Value

    @classmethod
    def calculate_portion(
        cls,
        fruit: FruitNutrition,
        quantity: float,
        unit: PortionUnit = PortionUnit.COUNT
    ) -> CalculatedPortion:
        """
        Calculates exact calories and complete nutritional profile.
        Formula:
          grams = count * avg_weight_per_piece_g  (when unit is count)
          calories = (grams / 100) * kcal_per_100g
        """
        if quantity <= 0:
            quantity = 1.0

        # 1. Determine exact weight in grams
        if unit == PortionUnit.COUNT:
            grams = quantity * fruit.avg_weight_per_piece_g
        elif unit == PortionUnit.GRAMS:
            grams = quantity
        elif unit == PortionUnit.OUNCES:
            grams = quantity * cls.GRAMS_PER_OUNCE
        elif unit == PortionUnit.CUPS:
            grams = quantity * cls.GRAMS_PER_CUP_SLICED
        else:
            grams = quantity * fruit.avg_weight_per_piece_g

        # 2. Scaling factor per 100g
        factor = grams / 100.0

        calories = round(factor * fruit.kcal_per_100g, 1)
        protein = round(factor * fruit.protein_g, 2)
        total_carbs = round(factor * fruit.carbs_g, 2)
        fiber = round(factor * fruit.fiber_g, 2)
        net_carbs = max(0.0, round(total_carbs - fiber, 2))
        sugars = round(factor * fruit.sugars_g, 2)
        fats = round(factor * fruit.fats_g, 2)
        water = round(factor * fruit.water_g, 1)
        vit_c = round(factor * fruit.vitamin_c_mg, 1)
        potassium = round(factor * fruit.potassium_mg, 1)

        vit_c_pct = round((vit_c / cls.VITAMIN_C_DAILY_VALUE_MG) * 100.0, 1)
        potassium_pct = round((potassium / cls.POTASSIUM_DAILY_VALUE_MG) * 100.0, 1)

        # 3. Glycemic Load calculation: GL = (GI * Net Carbs) / 100
        gl = round((fruit.glycemic_index * net_carbs) / 100.0, 1)
        if gl <= 10.0:
            gl_category = "Low"
        elif gl <= 19.0:
            gl_category = "Medium"
        else:
            gl_category = "High"

        return CalculatedPortion(
            fruit_name=fruit.fruit_name,
            input_quantity=quantity,
            unit=unit,
            calculated_grams=round(grams, 1),
            calories=calories,
            protein_g=protein,
            total_carbs_g=total_carbs,
            fiber_g=fiber,
            net_carbs_g=net_carbs,
            sugars_g=sugars,
            fats_g=fats,
            water_g=water,
            vitamin_c_mg=vit_c,
            vitamin_c_dv_pct=vit_c_pct,
            potassium_mg=potassium,
            potassium_dv_pct=potassium_pct,
            glycemic_index=fruit.glycemic_index,
            glycemic_load=gl,
            gl_category=gl_category,
            health_benefits=fruit.health_benefits,
            data_source=fruit.data_source
        )

    @classmethod
    def aggregate_items(cls, portions: List[CalculatedPortion]) -> AggregateNutritionReport:
        """
        Combines multiple fruit portions into an aggregate multi-fruit report.
        """
        total_grams = sum(p.calculated_grams for p in portions)
        total_calories = round(sum(p.calories for p in portions), 1)
        total_protein = round(sum(p.protein_g for p in portions), 2)
        total_carbs = round(sum(p.total_carbs_g for p in portions), 2)
        total_fiber = round(sum(p.fiber_g for p in portions), 2)
        total_net_carbs = round(sum(p.net_carbs_g for p in portions), 2)
        total_sugars = round(sum(p.sugars_g for p in portions), 2)
        total_fats = round(sum(p.fats_g for p in portions), 2)
        total_vit_c = round(sum(p.vitamin_c_mg for p in portions), 1)
        total_potassium = round(sum(p.potassium_mg for p in portions), 1)

        # Calorie breakdown by fruit
        cal_breakdown = {}
        for p in portions:
            pct = round((p.calories / total_calories * 100.0) if total_calories > 0 else 0, 1)
            cal_breakdown[p.fruit_name] = pct

        # Macro calories: Carbs 4 kcal/g, Protein 4 kcal/g, Fat 9 kcal/g
        carb_cals = total_carbs * 4.0
        prot_cals = total_protein * 4.0
        fat_cals = total_fats * 9.0
        macro_sum = carb_cals + prot_cals + fat_cals

        if macro_sum > 0:
            macro_dist = {
                "Carbohydrates": round((carb_cals / macro_sum) * 100.0, 1),
                "Protein": round((prot_cals / macro_sum) * 100.0, 1),
                "Fats": round((fat_cals / macro_sum) * 100.0, 1)
            }
        else:
            macro_dist = {"Carbohydrates": 100.0, "Protein": 0.0, "Fats": 0.0}

        return AggregateNutritionReport(
            items=portions,
            total_grams=round(total_grams, 1),
            total_calories=total_calories,
            total_protein_g=total_protein,
            total_carbs_g=total_carbs,
            total_fiber_g=total_fiber,
            total_net_carbs_g=total_net_carbs,
            total_sugars_g=total_sugars,
            total_fats_g=total_fats,
            total_vitamin_c_mg=total_vit_c,
            total_potassium_mg=total_potassium,
            calorie_breakdown_by_fruit=cal_breakdown,
            macro_distribution_pct=macro_dist
        )
