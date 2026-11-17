"""
FruitVision: Minimalist Multi-Fruit Analyzer & Calorie Tracking System
Architectural Swiss-style monochrome interface with subtle emerald/slate accents.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import altair as alt

# Ensure workspace and virtual environment are on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))

venv_site = str(BASE_DIR.parent / ".venv" / "Lib" / "site-packages")
if venv_site not in sys.path and Path(venv_site).exists():
    sys.path.insert(0, venv_site)

from fruit_analyzer.core.nutrition_db import NutritionDatabase, FruitNutrition
from fruit_analyzer.core.quantity_calculator import QuantityCalculator, PortionUnit, CalculatedPortion
from fruit_analyzer.core.tracker import DailyFruitTracker
from fruit_analyzer.models.classifier import FruitClassifier
from fruit_analyzer.models.detector import MultiFruitDetector


# Page Configuration
st.set_page_config(
    page_title="FruitVision - AI Food & Calorie Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist Clean White High-Contrast CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }

    /* Complete Light Theme Overrides - 100% White Clean Website Look */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    .main,
    section.main {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* Form widgets, inputs, select boxes, file uploader */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    input, textarea, select {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
    }

    /* Dataframe and Tables */
    [data-testid="stDataFrame"],
    .stTable,
    div[data-testid="stTable"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
    }

    /* Expanders and Tabs */
    [data-testid="stExpander"],
    details {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
    }

    /* High contrast text guarantees */
    p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
    }

    /* Brand Header */
    .brand-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 6px;
    }
    .brand-badge {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: #0F172A;
        color: #FFFFFF !important;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .brand-version {
        font-size: 0.75rem;
        color: #64748B !important;
        font-weight: 500;
    }
    .minimal-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #0F172A !important;
        margin: 0;
        line-height: 1.15;
    }
    .minimal-subtitle {
        color: #64748B !important;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 4px;
        margin-bottom: 1.8rem;
    }

    /* Minimalist Containers & Cards */
    .clean-card {
        background: #FAFAFA;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .metric-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 14px 16px;
        text-align: left;
        margin-bottom: 10px;
    }
    .metric-panel-num {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A !important;
        line-height: 1.1;
    }
    .metric-panel-label {
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B !important;
        margin-bottom: 6px;
    }
    .metric-panel-sub {
        font-size: 0.78rem;
        color: #94A3B8 !important;
        margin-top: 4px;
    }

    /* Tags & Status Pills */
    .mono-step {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #059669 !important;
        margin-bottom: 4px;
        display: block;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        letter-spacing: 0.02em;
    }
    .pill-accent {
        background: #ECFDF5;
        color: #065F46 !important;
        border: 1px solid #A7F3D0;
    }
    .pill-neutral {
        background: #F1F5F9;
        color: #334155 !important;
        border: 1px solid #E2E8F0;
    }
    .pill-warn {
        background: #FFFBEB;
        color: #92400E !important;
        border: 1px solid #FDE68A;
    }

    /* Clinical Note */
    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #10B981;
        border-radius: 6px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #334155 !important;
        line-height: 1.5;
        margin-top: 10px;
    }

    /* Buttons & Inputs */
    .stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        border: 1px solid #0F172A !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #1E293B !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }

    /* Clean Divider */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)


# State Initialization & Singletons
@st.cache_resource
def get_nutrition_db():
    return NutritionDatabase()

@st.cache_resource
def get_classifier():
    return FruitClassifier()

@st.cache_resource
def get_detector():
    return MultiFruitDetector(get_classifier())

nutrition_db = get_nutrition_db()
classifier = get_classifier()
detector = get_detector()

if "daily_tracker" not in st.session_state:
    st.session_state.daily_tracker = DailyFruitTracker()

if "confirmed_fruit" not in st.session_state:
    st.session_state.confirmed_fruit = None

if "deleted_instance_ids" not in st.session_state:
    st.session_state.deleted_instance_ids = set()

if "last_img_id" not in st.session_state:
    st.session_state.last_img_id = None

FRUIT_EMOJIS = {}

def configure_clean_white_chart(chart: alt.Chart) -> alt.Chart:
    """Ensures graphs are purely clean, white, and clearly readable without dark theme conflicts."""
    return (
        chart
        .properties(background="#FFFFFF")
        .configure_view(stroke=None, fill="#FFFFFF")
        .configure_axis(
            labelColor="#111827",
            titleColor="#111827",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="bold",
            gridColor="#F3F4F6",
            gridWidth=1,
            domainColor="#D1D5DB",
            domainWidth=1,
            tickColor="#D1D5DB",
            labelFont="Inter, system-ui, sans-serif",
            titleFont="Inter, system-ui, sans-serif"
        )
        .configure_legend(
            labelColor="#111827",
            titleColor="#111827",
            labelFontSize=11,
            titleFontSize=12,
            titleFontWeight="bold",
            fillColor="#FFFFFF",
            strokeColor="#E5E7EB",
            padding=8,
            cornerRadius=4,
            labelFont="Inter, system-ui, sans-serif",
            titleFont="Inter, system-ui, sans-serif"
        )
    )


# Sidebar
with st.sidebar:
    st.markdown('<div class="brand-badge">FRUITVISION</div>', unsafe_allow_html=True)
    st.markdown("### Navigation")

    app_mode = st.radio(
        "Select Mode",
        [
            "Multi-Fruit Detection",
            "Single-Fruit Analysis",
            "Nutrition Explorer",
            "Daily Intake Log"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<span class="mono-step">CALIBRATION</span>', unsafe_allow_html=True)
    st.markdown("**Confidence Threshold**")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.05,
        label_visibility="collapsed",
        help="Predictions below this confidence level prompt manual verification."
    )
    st.caption(f"Confirmation trigger: < {int(conf_threshold * 100)}%")

    st.markdown("---")
    st.markdown('<span class="mono-step">TEST SUITE</span>', unsafe_allow_html=True)
    st.markdown("**Load Preset Sample**")
    sample_dir = BASE_DIR / "data" / "sample_images"
    selected_sample = st.selectbox(
        "Preset Sample",
        [
            "None",
            "Apple (Single)",
            "Banana (Single)",
            "Orange (Single)",
            "Strawberry (Single)",
            "Watermelon (Single)",
            "Kiwi (Single)",
            "2 Bananas + 1 Apple",
            "Mixed Plate (4 Items)",
            "Mixed Basket (3 Items)"
        ],
        label_visibility="collapsed"
    )

    sample_image_path = None
    if selected_sample == "Apple (Single)":
        sample_image_path = sample_dir / "apple.jpg"
    elif selected_sample == "Banana (Single)":
        sample_image_path = sample_dir / "banana.jpg"
    elif selected_sample == "Orange (Single)":
        sample_image_path = sample_dir / "orange.jpg"
    elif selected_sample == "Strawberry (Single)":
        sample_image_path = sample_dir / "strawberry.jpg"
    elif selected_sample == "Watermelon (Single)":
        sample_image_path = sample_dir / "watermelon.jpg"
    elif selected_sample == "Kiwi (Single)":
        sample_image_path = sample_dir / "kiwi.jpg"
    elif selected_sample == "2 Bananas + 1 Apple":
        sample_image_path = sample_dir / "two_bananas_one_apple.jpg"
    elif selected_sample == "Mixed Plate (4 Items)":
        sample_image_path = sample_dir / "mixed_plate.jpg"
    elif selected_sample == "Mixed Basket (3 Items)":
        sample_image_path = sample_dir / "mixed_fruits.jpg"

    st.markdown("---")
    st.caption("Engine: MobileNetV3-Large Real-Stock ONNX | USDA SR Legacy")


# MODE 1: MULTI-FRUIT AUTO-DETECTOR & COUNTER
if app_mode == "Multi-Fruit Detection":
    st.markdown('<div class="brand-bar"><span class="brand-badge">VISION PIPELINE</span><span class="brand-version">Spatial Localization & Auto-Count</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="minimal-title">Multi-Fruit Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="minimal-subtitle">Segments multiple items in one image, localizes bounding boxes, and aggregates total nutritional intake.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    active_img = None

    with col1:
        st.markdown('<span class="mono-step">01 / INPUT IMAGE</span>', unsafe_allow_html=True)
        upload_choice = st.radio("Input Source", ["File Upload", "Camera"], horizontal=True, label_visibility="collapsed")

        if upload_choice == "File Upload":
            uploaded_file = st.file_uploader("Upload image containing fruits", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
            if uploaded_file:
                active_img = Image.open(uploaded_file)
            elif sample_image_path and sample_image_path.exists():
                active_img = Image.open(sample_image_path)
                st.caption(f"Preset loaded: **{selected_sample}**")
        else:
            cam_file = st.camera_input("Capture fruit photo", label_visibility="collapsed")
            if cam_file:
                active_img = Image.open(cam_file)

        if active_img:
            st.image(active_img, caption="Source Photo", use_container_width=True)

    with col2:
        st.markdown('<span class="mono-step">02 / LOCALIZATION & CLASSIFICATION</span>', unsafe_allow_html=True)
        if active_img is not None:
            # Check if image source changed to reset deleted items
            current_img_id = f"{upload_choice}_{uploaded_file.name if (upload_choice == 'File Upload' and uploaded_file) else selected_sample}"
            if st.session_state.last_img_id != current_img_id:
                st.session_state.last_img_id = current_img_id
                st.session_state.deleted_instance_ids = set()

            with st.spinner("Executing spatial decomposition..."):
                det_result = detector.detect(active_img)

            # Filter instances by active user selection (excluding deleted items)
            active_instances = [
                inst for inst in det_result.instances
                if inst.instance_id not in st.session_state.deleted_instance_ids
            ]

            # Re-render annotations showing ONLY active instances
            annotated_display = detector.render_annotations(active_img.copy(), active_instances)

            st.image(
                annotated_display,
                caption=f"Showing {len(active_instances)} Active Fruit(s) (of {det_result.total_detected} total detected)",
                use_container_width=True
            )

            if len(active_instances) == 0:
                st.warning("All detected items have been removed by user.")
                if st.button("Restore All Detected Items", key="restore_all_top"):
                    st.session_state.deleted_instance_ids.clear()
                    st.rerun()
            else:
                summary_badges = " ".join([
                    f"<span class='pill' style='background-color:{inst.color_hex}15; color:{inst.color_hex}; border:1px solid {inst.color_hex}; font-weight:700; margin-right:6px;'>"
                    f"#{inst.instance_id} {inst.fruit_name} ({int(inst.confidence*100)}%)</span>"
                    for inst in active_instances
                ])
                st.markdown(f"<div style='margin-top:8px;'>{summary_badges}</div>", unsafe_allow_html=True)

                if len(st.session_state.deleted_instance_ids) > 0:
                    del_n = len(st.session_state.deleted_instance_ids)
                    st.caption(f"{del_n} extra item(s) removed.")
                    if st.button("Restore All Deleted Items", key="restore_sub_btn"):
                        st.session_state.deleted_instance_ids.clear()
                        st.rerun()
        else:
            st.info("Select an image preset from the sidebar or upload a photo to start analysis.")

    # Breakdown & Aggregation Section
    if active_img is not None and len(active_instances) > 0:
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        sec_h1, sec_h2 = st.columns([3, 1])
        with sec_h1:
            st.markdown('<span class="mono-step">03 / PER-FRUIT CALORIE & PORTION ATTRIBUTION</span>', unsafe_allow_html=True)
            st.markdown("#### Individual Fruit Nutrition Breakdown")
            st.caption("Each card corresponds directly to the matching numbered bounding box above. Click 'Delete' to remove any false or extra detection.")
        with sec_h2:
            if len(st.session_state.deleted_instance_ids) > 0:
                if st.button("Restore All Items", key="restore_sec3"):
                    st.session_state.deleted_instance_ids.clear()
                    st.rerun()

        calculated_items = []
        item_cols = st.columns(min(3, max(1, len(active_instances))))

        for idx, inst in enumerate(active_instances):
            fruit_name = inst.fruit_name
            fruit_data = nutrition_db.get_fruit(fruit_name)
            if not fruit_data:
                fruit_data = nutrition_db.get_fruit("Apple")

            with item_cols[idx % len(item_cols)]:
                st.markdown(f"""
                <div class="clean-card" style="padding:14px; border-left: 4px solid {inst.color_hex}; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="background-color:{inst.color_hex}20; color:{inst.color_hex}; font-weight:700; font-size:0.80rem; padding:2px 8px; border-radius:4px; border:1px solid {inst.color_hex};">
                            Box #{inst.instance_id}
                        </span>
                        <span style="font-size:0.80rem; color:#6B7280;">{inst.confidence*100:.0f}% confidence</span>
                    </div>
                    <div style="font-weight:700; font-size:1.15rem; color:#111827;">
                        {fruit_name}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                u_col, q_col = st.columns([1, 1])
                with u_col:
                    user_unit = st.selectbox(
                        f"Unit #{inst.instance_id}",
                        [PortionUnit.COUNT.value, PortionUnit.GRAMS.value],
                        format_func=lambda x: "Piece / Count" if x == PortionUnit.COUNT.value else "Grams (g)",
                        key=f"det_unit_inst_{inst.instance_id}_{fruit_name}"
                    )
                with q_col:
                    if user_unit == PortionUnit.GRAMS.value:
                        default_g = float(round(fruit_data.avg_weight_per_piece_g, 1) if fruit_data and fruit_data.avg_weight_per_piece_g > 0 else 100.0)
                        user_qty = st.number_input(
                            f"Grams #{inst.instance_id}",
                            min_value=1.0,
                            max_value=5000.0,
                            value=default_g,
                            step=10.0,
                            key=f"det_qty_inst_{inst.instance_id}_{fruit_name}_g",
                            help=f"Weight in grams for Box #{inst.instance_id} (standard piece is {default_g:.0f}g)"
                        )
                    else:
                        user_qty = st.number_input(
                            f"Count #{inst.instance_id}",
                            min_value=0.25,
                            max_value=50.0,
                            value=1.0,
                            step=0.5,
                            key=f"det_qty_inst_{inst.instance_id}_{fruit_name}_cnt",
                            help=f"Number of pieces for Box #{inst.instance_id} ({fruit_name})"
                        )

                portion = QuantityCalculator.calculate_portion(fruit_data, user_qty, PortionUnit(user_unit))
                calculated_items.append((inst, portion))

                unit_label = f"{user_qty:.1f} pc" if user_unit == PortionUnit.COUNT.value else f"{user_qty:.0f}g"

                st.markdown(f"""
                <div style="background-color:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:10px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.85rem; color:#4B5563; font-weight:600;">{fruit_name} (#{inst.instance_id})</span>
                        <span style="background-color:#111827; color:#FFFFFF; font-weight:700; font-size:0.95rem; padding:3px 10px; border-radius:4px;">
                            {portion.calories} kcal
                        </span>
                    </div>
                    <div style="font-size:0.80rem; color:#374151; margin-bottom:4px;">
                        <b>Portion Weight:</b> {portion.calculated_grams}g ({unit_label})
                    </div>
                    <div style="font-size:0.78rem; color:#6B7280; line-height:1.4;">
                        <b>Carbs:</b> {portion.net_carbs_g}g • <b>Fiber:</b> {portion.fiber_g}g • <b>Sugars:</b> {portion.sugars_g}g • <b>Protein:</b> {portion.protein_g}g
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Delete Item #{inst.instance_id}", key=f"del_btn_inst_{inst.instance_id}", help=f"Remove Box #{inst.instance_id} if falsely detected"):
                    st.session_state.deleted_instance_ids.add(inst.instance_id)
                    st.rerun()

        portions_only = [p for _, p in calculated_items]
        agg_report = QuantityCalculator.aggregate_items(portions_only)

        # Explicit Structured Itemized Table
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("##### Detailed Nutrition Attribution Table")
        table_rows = []
        for inst, p in calculated_items:
            table_rows.append({
                "Bounding Box": f"Box #{inst.instance_id}",
                "Identified Fruit": inst.fruit_name,
                "Portion Size": f"{p.input_quantity} {p.unit.value} ({p.calculated_grams}g)",
                "Calories (kcal)": p.calories,
                "Net Carbs (g)": p.net_carbs_g,
                "Fiber (g)": p.fiber_g,
                "Sugars (g)": p.sugars_g,
                "Protein (g)": p.protein_g,
                "Fat (g)": p.fats_g
            })

        # Add Summary Total Row
        table_rows.append({
            "Bounding Box": "TOTAL MEAL",
            "Identified Fruit": f"{len(calculated_items)} Items Combined",
            "Portion Size": f"{agg_report.total_grams}g Total",
            "Calories (kcal)": agg_report.total_calories,
            "Net Carbs (g)": agg_report.total_net_carbs_g,
            "Fiber (g)": agg_report.total_fiber_g,
            "Sugars (g)": agg_report.total_sugars_g,
            "Protein (g)": agg_report.total_protein_g,
            "Fat (g)": agg_report.total_fats_g
        })
        summary_table_df = pd.DataFrame(table_rows)
        st.dataframe(summary_table_df, use_container_width=True, hide_index=True)

        # Aggregate Meal Metrics
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<span class="mono-step">04 / MEAL TOTALS & SIMPLE CHARTS</span>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Total Calories</div>
                <div class="metric-panel-num">{agg_report.total_calories}</div>
                <div class="metric-panel-sub">kcal</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Total Mass</div>
                <div class="metric-panel-num">{agg_report.total_grams}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Net Carbs</div>
                <div class="metric-panel-num">{agg_report.total_net_carbs_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Dietary Fiber</div>
                <div class="metric-panel-num">{agg_report.total_fiber_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Natural Sugar</div>
                <div class="metric-panel-num">{agg_report.total_sugars_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with m6:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Protein</div>
                <div class="metric-panel-num">{agg_report.total_protein_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)

        # Simple Graphs: Combined Calorie Pie Chart & Colorful Macronutrient Bar Graph
        c_col1, c_col2 = st.columns([1, 1], gap="large")

        with c_col1:
            st.markdown("**Calorie Breakdown & Share by Fruit (Pie Chart)**")
            
            tot_cals = max(1, agg_report.total_calories)
            cal_pie_data = []
            for inst, p in calculated_items:
                pct = round((p.calories / tot_cals) * 100, 1)
                cal_pie_data.append({
                    "Item": f"#{inst.instance_id} {inst.fruit_name}",
                    "Calories (kcal)": p.calories,
                    "Share (%)": pct,
                    "Label": f"#{inst.instance_id} {inst.fruit_name}: {p.calories} kcal ({pct:.0f}%)",
                    "Portion": f"{p.input_quantity} {p.unit.value} ({p.calculated_grams}g)"
                })
            cal_pie_df = pd.DataFrame(cal_pie_data)

            # Colorful, modern, non-monotone palette for distinct fruit items
            pie_palette = [
                "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
                "#06B6D4", "#EC4899", "#F97316", "#14B8A6", "#6366F1"
            ]

            pie_chart = (
                alt.Chart(cal_pie_df)
                .mark_arc(outerRadius=95, stroke="#FFFFFF", strokeWidth=2)
                .encode(
                    theta=alt.Theta(field="Calories (kcal)", type="quantitative"),
                    color=alt.Color(
                        field="Label",
                        type="nominal",
                        scale=alt.Scale(range=pie_palette[:len(cal_pie_df)]),
                        legend=alt.Legend(
                            title="Detected Fruits (Calories & Share)",
                            orient="bottom",
                            columns=1,
                            labelFontSize=11,
                            titleFontSize=11,
                            titleFontWeight="bold"
                        )
                    ),
                    tooltip=["Item", "Calories (kcal)", "Share (%)", "Portion"]
                )
                .properties(height=280, background="#FFFFFF")
            )
            clean_pie = configure_clean_white_chart(pie_chart)
            st.markdown('<div class="clean-card" style="background:#FFFFFF; padding:16px; border:1px solid #E2E8F0; border-radius:8px;">', unsafe_allow_html=True)
            st.altair_chart(clean_pie, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_col2:
            st.markdown("**Meal Macronutrients Breakdown (Bar Graph)**")
            macro_palette = {
                "Carbohydrates": "#3B82F6",
                "Dietary Fiber": "#10B981",
                "Natural Sugars": "#F59E0B",
                "Protein": "#8B5CF6",
                "Fats": "#EC4899"
            }
            macro_df = pd.DataFrame([
                {"Nutrient": "Carbohydrates", "Amount (g)": agg_report.total_net_carbs_g},
                {"Nutrient": "Dietary Fiber", "Amount (g)": agg_report.total_fiber_g},
                {"Nutrient": "Natural Sugars", "Amount (g)": agg_report.total_sugars_g},
                {"Nutrient": "Protein", "Amount (g)": agg_report.total_protein_g},
                {"Nutrient": "Fats", "Amount (g)": agg_report.total_fats_g},
            ])
            macro_bar = (
                alt.Chart(macro_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Nutrient:N", title=None, axis=alt.Axis(labelAngle=0, labelFontWeight="bold")),
                    y=alt.Y("Amount (g):Q", title="Weight (grams)", axis=alt.Axis(grid=True, gridColor="#F1F5F9")),
                    color=alt.Color(
                        "Nutrient:N",
                        scale=alt.Scale(
                            domain=list(macro_palette.keys()),
                            range=list(macro_palette.values())
                        ),
                        legend=None
                    ),
                    tooltip=["Nutrient", "Amount (g)"]
                )
            )
            macro_text = macro_bar.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                fontSize=11,
                fontWeight="bold",
                color="#0F172A"
            ).encode(
                text=alt.Text("Amount (g):Q")
            )
            clean_macro = configure_clean_white_chart((macro_bar + macro_text).properties(height=280, background="#FFFFFF"))
            st.markdown('<div class="clean-card" style="background:#FFFFFF; padding:16px; border:1px solid #E2E8F0; border-radius:8px;">', unsafe_allow_html=True)
            st.altair_chart(clean_macro, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

        # Log Action
        if st.button("Log Meal to Daily Intake", type="primary", use_container_width=True):
            for _, p in calculated_items:
                st.session_state.daily_tracker.log_portion(p)
            st.success(f"Successfully recorded {len(calculated_items)} items ({agg_report.total_calories} kcal) to today's log.")


# MODE 2: SINGLE-FRUIT ANALYSIS
elif app_mode == "Single-Fruit Analysis":
    st.markdown('<div class="brand-bar"><span class="brand-badge">HIGH RESOLUTION</span><span class="brand-version">Morphometric Inference</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="minimal-title">Single-Fruit Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="minimal-subtitle">Fine-grained identification with probability distributions and clinical glycemic load scoring.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    active_img = None

    with col1:
        st.markdown('<span class="mono-step">01 / SOURCE PHOTO</span>', unsafe_allow_html=True)
        up_choice = st.radio("Source", ["File Upload", "Camera"], horizontal=True, label_visibility="collapsed", key="single_up_src")

        if up_choice == "File Upload":
            uploaded = st.file_uploader("Upload single fruit photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed", key="s_file")
            if uploaded:
                active_img = Image.open(uploaded)
            elif sample_image_path and sample_image_path.exists():
                active_img = Image.open(sample_image_path)
                st.caption(f"Preset loaded: **{selected_sample}**")
        else:
            cam = st.camera_input("Capture fruit photo", label_visibility="collapsed", key="s_cam")
            if cam:
                active_img = Image.open(cam)

        if active_img:
            st.image(active_img, caption="Input Photo", use_container_width=True)

    with col2:
        st.markdown('<span class="mono-step">02 / PREDICTION & PROBABILITY</span>', unsafe_allow_html=True)
        if active_img is not None:
            pred = classifier.predict(active_img, confidence_threshold=conf_threshold)
            target_name = st.session_state.confirmed_fruit or pred.top_fruit

            pill_cls = "pill-accent" if not pred.is_low_confidence else "pill-warn"
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <span style="font-size:1.75rem; font-weight:700; color:#111827; margin-right:12px;">{target_name}</span>
                <span class="pill {pill_cls}">{pred.confidence * 100:.1f}% Confidence</span>
            </div>
            """, unsafe_allow_html=True)

            # Low Confidence Disambiguation
            if pred.is_low_confidence:
                st.warning(f"Confidence is below configured threshold ({int(conf_threshold*100)}%). Verify selection:")
                candidates = [c[0] for c in pred.top_5]
                chosen = st.selectbox("Confirm Identification:", candidates, index=0)
                if st.button("Confirm Choice"):
                    st.session_state.confirmed_fruit = chosen
                    st.rerun()

            # Probability Distribution Chart
            st.markdown("**Top Candidate Probabilities**")
            prob_df = pd.DataFrame([
                {"Fruit": f, "Probability (%)": round(p * 100, 1)} for f, p in pred.top_5
            ])
            prob_palette = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]
            prob_chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=True, gridColor="#F1F5F9")),
                    y=alt.Y("Fruit:N", sort="-x", axis=alt.Axis(title=None)),
                    color=alt.Color("Fruit:N", scale=alt.Scale(range=prob_palette), legend=None),
                    tooltip=["Fruit", "Probability (%)"]
                )
            )
            clean_prob = configure_clean_white_chart(prob_chart.properties(height=160, background="#FFFFFF"))
            st.altair_chart(clean_prob, use_container_width=True, theme=None)

            if st.session_state.confirmed_fruit and st.session_state.confirmed_fruit != pred.top_fruit:
                if st.button("Reset to Automated Model Prediction"):
                    st.session_state.confirmed_fruit = None
                    st.rerun()
        else:
            st.info("Upload a fruit photo or pick a sample preset to view predictions.")

    # Single Fruit Nutrition Calculation
    if active_img is not None:
        target_fruit_name = st.session_state.confirmed_fruit or pred.top_fruit
        fruit_data = nutrition_db.get_fruit(target_fruit_name)
        if not fruit_data:
            fruit_data = nutrition_db.get_fruit("Apple")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<span class="mono-step">03 / PORTION WEIGHT & ENERGY CALCULATION</span>', unsafe_allow_html=True)

        q1, q2, q3 = st.columns([1, 1, 1], gap="medium")
        with q1:
            q_unit = st.selectbox(
                "Portion Unit",
                [PortionUnit.COUNT.value, PortionUnit.GRAMS.value],
                format_func=lambda x: "Whole Piece / Count" if x == PortionUnit.COUNT.value else "Grams (g)"
            )
        with q2:
            if q_unit == PortionUnit.GRAMS.value:
                default_q = float(round(fruit_data.avg_weight_per_piece_g, 1) if fruit_data and fruit_data.avg_weight_per_piece_g > 0 else 100.0)
                q_val = st.number_input("Weight in Grams", min_value=1.0, max_value=5000.0, value=default_q, step=10.0, key="single_qty_grams")
            else:
                default_q = 1.0
                q_val = st.number_input("Piece Count", min_value=0.25, max_value=50.0, value=1.0, step=0.5, key="single_qty_count")

        portion = QuantityCalculator.calculate_portion(fruit_data, q_val, PortionUnit(q_unit))

        with q3:
            st.markdown(f"""
            <div class="metric-panel" style="margin-top:0;">
                <div class="metric-panel-label">Calculated Mass</div>
                <div class="metric-panel-num">{portion.calculated_grams} <span style="font-size:1rem; font-weight:500;">g</span></div>
                <div class="metric-panel-sub">Avg piece reference: {fruit_data.avg_weight_per_piece_g}g</div>
            </div>
            """, unsafe_allow_html=True)

        # KPI Metrics
        st.markdown("#### Nutritional Profile")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Calories</div>
                <div class="metric-panel-num">{portion.calories}</div>
                <div class="metric-panel-sub">kcal</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Net Carbs</div>
                <div class="metric-panel-num">{portion.net_carbs_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Dietary Fiber</div>
                <div class="metric-panel-num">{portion.fiber_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Sugars</div>
                <div class="metric-panel-num">{portion.sugars_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Protein</div>
                <div class="metric-panel-num">{portion.protein_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)
        with k6:
            st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-panel-label">Fats</div>
                <div class="metric-panel-num">{portion.fats_g}</div>
                <div class="metric-panel-sub">grams</div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical Insights Card
        st.markdown(f"""
        <div class="insight-card">
            <span class="mono-step">CLINICAL & HEALTH METRICS</span>
            <div style="margin-top:4px;"><b>Glycemic Load:</b> {portion.glycemic_load} ({portion.gl_category}) • <b>Glycemic Index:</b> {portion.glycemic_index} ({fruit_data.gi_category})</div>
            <div style="margin-top:4px; color:#4B5563;">{portion.health_benefits}</div>
        </div>
        """, unsafe_allow_html=True)

        # Simple Graphs: Macronutrients Bar Graph & Energy Ratio Pie Graph
        st.markdown("<br/>", unsafe_allow_html=True)
        s_g1, s_g2 = st.columns([1, 1], gap="large")
        with s_g1:
            st.markdown("**Macronutrients (Simple Bar Graph)**")
            s_macro_palette = {
                "Net Carbs": "#3B82F6",
                "Fiber": "#10B981",
                "Sugars": "#F59E0B",
                "Protein": "#8B5CF6",
                "Fats": "#EC4899"
            }
            s_macro_df = pd.DataFrame([
                {"Nutrient": "Net Carbs", "Grams": portion.net_carbs_g},
                {"Nutrient": "Fiber", "Grams": portion.fiber_g},
                {"Nutrient": "Sugars", "Grams": portion.sugars_g},
                {"Nutrient": "Protein", "Grams": portion.protein_g},
                {"Nutrient": "Fats", "Grams": portion.fats_g},
            ])
            s_m_bar = (
                alt.Chart(s_macro_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Nutrient:N", title=None, axis=alt.Axis(labelAngle=0, labelFontWeight="bold")),
                    y=alt.Y("Grams:Q", title="Weight (g)", axis=alt.Axis(grid=True, gridColor="#F1F5F9")),
                    color=alt.Color(
                        "Nutrient:N",
                        scale=alt.Scale(
                            domain=list(s_macro_palette.keys()),
                            range=list(s_macro_palette.values())
                        ),
                        legend=None
                    ),
                    tooltip=["Nutrient", "Grams"]
                )
            )
            s_m_text = s_m_bar.mark_text(
                align="center",
                baseline="bottom",
                dy=-4,
                fontSize=11,
                fontWeight="bold",
                color="#0F172A"
            ).encode(
                text=alt.Text("Grams:Q")
            )
            clean_sm = configure_clean_white_chart((s_m_bar + s_m_text).properties(height=240, background="#FFFFFF"))
            st.markdown('<div class="clean-card" style="background:#FFFFFF; padding:14px; border:1px solid #E2E8F0; border-radius:8px;">', unsafe_allow_html=True)
            st.altair_chart(clean_sm, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

        with s_g2:
            st.markdown("**Energy Calorie Split (Simple Pie Graph)**")
            cal_carb = round(portion.net_carbs_g * 4.0, 1)
            cal_prot = round(portion.protein_g * 4.0, 1)
            cal_fat = round(portion.fats_g * 9.0, 1)
            s_pie_df = pd.DataFrame([
                {"Source": "Carbohydrates", "Calories": max(0.1, cal_carb)},
                {"Source": "Protein", "Calories": max(0.1, cal_prot)},
                {"Source": "Fats", "Calories": max(0.1, cal_fat)},
            ])
            s_pie = alt.Chart(s_pie_df).mark_arc(outerRadius=85, stroke="#FFFFFF", strokeWidth=2).encode(
                theta=alt.Theta(field="Calories", type="quantitative"),
                color=alt.Color(
                    field="Source",
                    type="nominal",
                    scale=alt.Scale(
                        domain=["Carbohydrates", "Protein", "Fats"],
                        range=["#3B82F6", "#8B5CF6", "#EC4899"]
                    ),
                    legend=alt.Legend(title="Macro Energy Source", orient="bottom")
                ),
                tooltip=["Source", "Calories"]
            ).properties(height=240, background="#FFFFFF")
            clean_spie = configure_clean_white_chart(s_pie)
            st.markdown('<div class="clean-card" style="background:#FFFFFF; padding:14px; border:1px solid #E2E8F0; border-radius:8px;">', unsafe_allow_html=True)
            st.altair_chart(clean_spie, use_container_width=True, theme=None)
            st.markdown('</div>', unsafe_allow_html=True)

        # Micronutrient Progress
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("**Micronutrient Density**")
        u1, u2, u3 = st.columns(3)
        with u1:
            st.progress(min(1.0, portion.vitamin_c_dv_pct / 100.0))
            st.caption(f"Vitamin C: **{portion.vitamin_c_mg} mg** ({portion.vitamin_c_dv_pct}% Daily Value)")
        with u2:
            st.progress(min(1.0, portion.potassium_dv_pct / 100.0))
            st.caption(f"Potassium: **{portion.potassium_mg} mg** ({portion.potassium_dv_pct}% Daily Value)")
        with u3:
            st.progress(min(1.0, portion.water_g / max(1.0, portion.calculated_grams)))
            st.caption(f"Hydration: **{portion.water_g} g** water ({int(portion.water_g / max(1.0, portion.calculated_grams) * 100)}% content)")

        if st.button("Log to Daily Intake", type="primary", use_container_width=True):
            st.session_state.daily_tracker.log_portion(portion)
            st.success(f"Recorded {portion.input_quantity} {portion.unit.value} of {portion.fruit_name} ({portion.calories} kcal).")


# MODE 3: NUTRITION EXPLORER
elif app_mode == "Nutrition Explorer":
    st.markdown('<div class="brand-bar"><span class="brand-badge">DATABASE</span><span class="brand-version">USDA FoodData Central SR Legacy</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="minimal-title">Nutrition Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="minimal-subtitle">Search calibrated reference datasets for 40+ fruit varieties or query live USDA endpoints.</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Curated Fruit Database", "Live USDA API Search"])

    with tab1:
        all_fruits = nutrition_db.get_all_records()
        records = [{
            "Fruit": f.fruit_name,
            "Category": f.category,
            "Energy (kcal/100g)": f.kcal_per_100g,
            "Unit Weight (g)": f.avg_weight_per_piece_g,
            "Carbohydrates (g)": f.carbs_g,
            "Fiber (g)": f.fiber_g,
            "Sugar (g)": f.sugars_g,
            "Protein (g)": f.protein_g,
            "GI": f.glycemic_index,
            "GI Rating": f.gi_category,
            "USDA FDC ID": f.usda_fdc_id
        } for f in all_fruits.values()]

        df = pd.DataFrame(records)

        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            q_search = st.text_input("Filter fruit name or alias:", placeholder="e.g. apple, fuji, mandarin, banana...")
        with f_col2:
            cat_choice = st.selectbox("Category", ["All"] + sorted(list(df["Category"].unique())))

        filtered = df
        if q_search:
            filtered = filtered[filtered["Fruit"].str.contains(q_search, case=False, na=False)]
        if cat_choice != "All":
            filtered = filtered[filtered["Category"] == cat_choice]

        st.dataframe(filtered, use_container_width=True, height=450)

    with tab2:
        st.markdown("**Live USDA FoodData Central Query**")
        usda_term = st.text_input("Fruit Query:", placeholder="e.g. Dragonfruit, Guava, Kumquat")
        api_key = st.text_input("API Key (leave default for demo):", value="DEMO_KEY", type="password")

        if st.button("Query USDA Database"):
            if usda_term:
                with st.spinner("Fetching official records..."):
                    res = nutrition_db.fetch_from_usda_api(usda_term, api_key=api_key)
                if res:
                    st.success(f"Matched record: **{res.fruit_name}** ({res.data_source})")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Calories", f"{res.kcal_per_100g} kcal/100g")
                    with c2:
                        st.metric("Carbohydrates", f"{res.carbs_g} g")
                    with c3:
                        st.metric("Dietary Fiber", f"{res.fiber_g} g")
                    with c4:
                        st.metric("Protein", f"{res.protein_g} g")
                else:
                    st.info("USDA API unavailable or fruit not found. Using local calibrated database.")



# MODE 4: DAILY INTAKE LOG
elif app_mode == "Daily Intake Log":
    st.markdown('<div class="brand-bar"><span class="brand-badge">LOGGING</span><span class="brand-version">WHO 400g/Day Guideline Tracking</span></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="minimal-title">Daily Intake Log</h1>', unsafe_allow_html=True)
    st.markdown('<p class="minimal-subtitle">Track cumulative caloric intake and dietary fiber progress against international nutrition standards.</p>', unsafe_allow_html=True)

    summary = st.session_state.daily_tracker.get_summary()

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Logged Items</div>
            <div class="metric-panel-num">{summary['total_items']}</div>
            <div class="metric-panel-sub">entries today</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Mass</div>
            <div class="metric-panel-num">{summary['total_grams']}</div>
            <div class="metric-panel-sub">grams consumed</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Total Energy</div>
            <div class="metric-panel-num">{summary['total_calories']}</div>
            <div class="metric-panel-sub">kcal</div>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown(f"""
        <div class="metric-panel">
            <div class="metric-panel-label">Dietary Fiber</div>
            <div class="metric-panel-num">{summary['total_fiber_g']}</div>
            <div class="metric-panel-sub">grams</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("**WHO 400g/Day Target Progress**")
    st.progress(summary["who_progress_pct"] / 100.0)
    st.caption(f"**{summary['total_grams']}g** of recommended **400g/day** reached ({summary['who_progress_pct']}%).")

    st.markdown("<hr/>", unsafe_allow_html=True)
    log_df = st.session_state.daily_tracker.to_dataframe()

    if not log_df.empty:
        st.dataframe(log_df, use_container_width=True)
        d1, d2 = st.columns([1, 1])
        with d1:
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export CSV Report", data=csv, file_name="daily_fruit_log.csv", mime="text/csv", use_container_width=True)
        with d2:
            if st.button("Clear Today's Log", use_container_width=True):
                st.session_state.daily_tracker.clear()
                st.rerun()
    else:
        st.info("No meals logged yet today. Use Multi-Fruit Detection or Single-Fruit Analysis to record your consumption.")
