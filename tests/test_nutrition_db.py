import pytest
from fruit_analyzer.core.nutrition_db import NutritionDatabase, FruitNutrition


@pytest.fixture
def db():
    return NutritionDatabase()


def test_db_loading(db):
    fruits = db.list_all_fruits()
    assert len(fruits) >= 40
    assert "Apple" in fruits
    assert "Banana" in fruits
    assert "Orange" in fruits


def test_exact_lookup(db):
    apple = db.get_fruit("Apple")
    assert apple is not None
    assert apple.fruit_name == "Apple"
    assert apple.kcal_per_100g == 52.0
    assert apple.avg_weight_per_piece_g == 182.0


def test_alias_lookup(db):
    # Granny smith is an alias for Apple
    apple = db.get_fruit("granny smith")
    assert apple is not None
    assert apple.fruit_name == "Apple"

    # Mandarin is an alias for Orange
    orange = db.get_fruit("mandarin")
    assert orange is not None
    assert orange.fruit_name == "Orange"

    # Amla is an alias for Gooseberry
    amla = db.get_fruit("amla")
    assert amla is not None
    assert amla.fruit_name == "Gooseberry"


def test_fuzzy_matching(db):
    # Slight typos
    b = db.get_fruit("bananna")
    assert b is not None
    assert b.fruit_name == "Banana"

    strw = db.get_fruit("strawbery")
    assert strw is not None
    assert strw.fruit_name == "Strawberry"
