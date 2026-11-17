# 🍎 FruitVision: Intelligent Multi-Fruit Analyzer & Calorie Tracking System

An end-to-end, production-grade vision AI application that identifies single or multiple fruits from photos, automatically counts instances with spatial bounding boxes, computes exact calorie and macronutrient breakdowns from authoritative USDA-calibrated databases, and tracks daily nutritional goals.

---

## 🌟 Key Features (Exceeding the Roadmap)

### 1. 🍇 Multi-Fruit Auto-Detection & Spatial Counting
- **Simultaneous Multi-Item Analysis**: Detects multiple fruits on a single plate, table, or bowl.
- **Visual Bounding Boxes**: Automatically draws color-coded, labeled bounding boxes with instance IDs (`#1 Apple 85%`, `#2 Banana 88%`).
- **Auto-Counting**: Computes item counts per fruit automatically and populates editable portion cards.
- **Aggregate Plate Totals**: Computes cumulative calories, net carbs, dietary fiber, sugar, and macro distributions with interactive Altair charts.

### 2. 🍎 Single-Fruit Deep Classifier & Guardrails
- **MobileNetV2 ONNX Engine**: Zero-dependency local CPU inference (~15ms per frame) with INT8 quantization.
- **Top-5 Probability Breakdown**: Displays visual probability bars for candidate predictions.
- **Low-Confidence Disambiguation & Confirmation**: If confidence falls below the user-set threshold (e.g. 65%), the app displays a prominent confirmation prompt with one-click candidate confirmation buttons or manual dropdown selection.

### 3. ⚖️ Flexible Quantity & Macro Engine
- Accepts **Count (Pieces)**, **Grams (g)**, **Ounces (oz)**, or **Sliced Cups**.
- **Exact Caloric Formula**:
  $$\text{grams} = \text{count} \times \text{avg\_weight\_per\_piece\_g}$$
  $$\text{kcal} = \left(\frac{\text{grams}}{100}\right) \times \text{kcal\_per\_100g}$$
- **Nutritional Depth**:
  - Net Carbs ($\max(0, \text{Total Carbs} - \text{Fiber})$)
  - Glycemic Index (GI) and Glycemic Load (GL) calculation:
    $$\text{GL} = \frac{\text{GI} \times \text{Net Carbs}}{100}$$
  - Micronutrients: Vitamin C (% Daily Value), Potassium (% Daily Value), Hydration percentage.
  - Scientific clinical health insights for each fruit.

### 4. 📚 Comprehensive Nutrition Database & Live USDA Sync
- **Curated 40+ Fruits CSV**: Calibrated against USDA SR Legacy and Foundation foods with official `usda_fdc_id` provenance.
- **RapidFuzz Synonym & Alias Resolution**: Automatically maps varieties like "Granny Smith" $\to$ Apple, "Mandarin" $\to$ Orange, "Amla" $\to$ Gooseberry, "Anjeer" $\to$ Fig.
- **Live USDA API Integration**: Integrated `fetch_from_usda_api()` querying USDA FoodData Central with local SQLite caching.

### 5. 🎯 Google Colab MobileNetV2 Fine-Tuning Pipeline
- Standalone Python training script: `fine_tuning/train_mobilenet_fruits360.py`
- Interactive Google Colab Notebook: `fine_tuning/Fruit_Classifier_MobileNetV2_Colab.ipynb`
- Features **rotation, color jitter, blur, and affine augmentations**, two-stage warmup + fine-tuning, dynamic INT8 ONNX quantization, and TFLite export.

### 6. 📊 Daily Fruit Intake Tracker
- Log meals across a session.
- Monitors progress against the **World Health Organization (WHO) 400g/day fruit and vegetable guideline**.
- One-click export to CSV (`daily_fruit_intake.csv`).

---

## 🏗️ Directory Architecture

```
fruit_analyzer/
├── app.py                      # Interactive Streamlit Web Application
├── run_app.bat                 # 1-click Windows launcher
├── requirements.txt            # Application dependencies
├── README.md                   # System documentation
├── core/
│   ├── __init__.py
│   ├── nutrition_db.py         # Nutrition DB (CSV + USDA API + Cache + RapidFuzz)
│   ├── quantity_calculator.py  # Portion math, unit conversions, macros & GL
│   └── tracker.py              # Session intake logger & WHO goal tracking
├── models/
│   ├── __init__.py
│   ├── classifier.py           # MobileNetV2 ONNX & Hugging Face vision wrapper
│   ├── detector.py             # Multi-fruit spatial segmentation & bounding box counter
│   └── weights/                # Stored ONNX models & configs
│       ├── mobilenet_v2_quantized.onnx
│       └── mobilenet_v2_config.json
├── data/
│   ├── fruit_nutrition.csv     # 40+ fruits database with macros, vitamins, GI
│   └── sample_images/          # Built-in test samples (apple, banana, orange, mixed)
├── fine_tuning/
│   ├── README.md               # Step-by-step Colab fine-tuning instructions
│   ├── Fruit_Classifier_MobileNetV2_Colab.ipynb # Free GPU Colab Notebook
│   └── train_mobilenet_fruits360.py             # Standalone Python script
└── tests/
    ├── test_nutrition_db.py    # Unit tests for database & alias matching
    ├── test_quantity_logic.py  # Unit tests for count/gram conversions & calories
    ├── test_classifier.py      # Unit tests for classifier, detector & thresholds
    └── test_tracker.py         # Unit tests for intake logger
```

---

## 🚀 Quickstart Guide

### 1. Launch Web Application
Double-click `run_app.bat` or run from terminal:
```bash
python -m streamlit run fruit_analyzer/app.py
```
Open `http://localhost:8501` in your browser.

### 2. Run Automated Test Suite
```bash
python -c "import sys; sys.path.insert(0, r'.\.venv\Lib\site-packages'); import pytest; sys.exit(pytest.main(['fruit_analyzer/tests', '-v']))"
```
All 14 unit and integration tests will execute and validate the pipeline.
