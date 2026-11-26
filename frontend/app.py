import os
import joblib
import pandas as pd
import streamlit as st

# 1. Load your saved model and label encoder from 'saved_models/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

model = joblib.load(os.path.join(MODEL_DIR, "xgb_nutriscore_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# 2. Page configuration
st.set_page_config(page_title="Bite Metric", page_icon="🥗", layout="centered")

# Initialize default values in session_state for the Reset button
DEFAULTS = {
    "energy": 1000.0,
    "sugars": 5.0,
    "saturated_fat": 2.0,
    "sodium": 0.5,
    "fiber": 3.0,
    "proteins": 8.0,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_inputs():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# Custom CSS
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #F5F1E3;
        color: #1B4332;
    }

    /* Balanced top padding so the title sits high without getting clipped */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 4.5rem !important;
    }

    /* Match top header bar to page background */
    header[data-testid="stHeader"] {
        background-color: #F5F1E3 !important;
    }
    header[data-testid="stHeader"] * {
        color: #1B4332 !important;
    }

    /* Global text & label colors */
    h1, h2, h3, p, label, .stMarkdown {
        color: #1B4332 !important;
    }

    /* Form container */
    [data-testid="stForm"] {
        background-color: #FAF7EE;
        border: 1px solid #D8D2C0;
        border-radius: 12px;
        padding: 24px;
    }

    /* Force ALL inner wrapper layers of the number input box to dark green */
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] > div {
        background-color: #1B4332 !important;
        border-color: #1B4332 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Keep the - and + controls */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background-color: #2D6A4F !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background-color: #40916C !important;
    }
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {
        fill: #FFFFFF !important;
    }

    /* Form submit buttons (Predict & Reset) */
    [data-testid="stFormSubmitButton"] > button {
        background-color: #2D6A4F !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 0.55rem 1rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stFormSubmitButton"] > button p,
    [data-testid="stFormSubmitButton"] > button span {
        color: #FFFFFF !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1B4332 !important;
    }

    /* Fixed bottom-left social links in dark green */
    .bottom-left-links {
        position: fixed;
        bottom: 16px;
        left: 24px;
        z-index: 9999;
        display: flex;
        gap: 16px;
        background-color: #1B4332;
        padding: 8px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
    }
    .bottom-left-links a {
        color: #FFFFFF !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
    }
    .bottom-left-links a:hover {
        color: #B7E4C7 !important;
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered title
st.markdown(
    "<h1 style='text-align: center; color: #1B4332; margin-top: 0px; margin-bottom: 0px;'>🥗 Bite Metric</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: #1B4332; margin-top: 0px; margin-bottom: 0px;'>From food composition to predictive nutritional scoring.</h5>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #1B4332; margin-bottom: 24px;'> Enter the nutritional composition per 100g and let our machine learning model <br> predict its Nutri-Score grade (A–E).</p>",
    unsafe_allow_html=True
)

# 3. Nutrient input form
with st.form("nutriscore_form"):
    col1, col2 = st.columns(2)

    with col1:
        energy = st.number_input("Energy (kJ per 100g)", min_value=0.0, max_value=4000.0, key="energy")
        sugars = st.number_input("Sugars (g)", min_value=0.0, max_value=100.0, key="sugars")
        saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, max_value=100.0, key="saturated_fat")

    with col2:
        sodium = st.number_input("Sodium (g)", min_value=0.0, max_value=40.0, key="sodium")
        fiber = st.number_input("Fiber (g)", min_value=0.0, max_value=100.0, key="fiber")
        proteins = st.number_input("Proteins (g)", min_value=0.0, max_value=100.0, key="proteins")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        submitted = st.form_submit_button("Predict Nutri-Score")
    with btn_col2:
        reset = st.form_submit_button("Reset", on_click=reset_inputs)

# 4. Compute engineered features & run prediction
if submitted:
    sugar_to_fiber_ratio = sugars / (fiber + 1.0)
    protein_to_fat_ratio = proteins / (saturated_fat + 1.0)

    input_df = pd.DataFrame([{
        "energy_100g": energy,
        "sugars_100g": sugars,
        "saturated-fat_100g": saturated_fat,
        "sodium_100g": sodium,
        "fiber_100g": fiber,
        "proteins_100g": proteins,
        "sugar_to_fiber_ratio": sugar_to_fiber_ratio,
        "protein_to_fat_ratio": protein_to_fat_ratio
    }])

    pred_encoded = model.predict(input_df)
    grade = label_encoder.inverse_transform(pred_encoded)[0].upper()

    grade_info = {
        "A": (
            "#038141",
            "Wholesome & Nutrient-Dense — Packed with body-fueling fiber and protein while keeping sugars and unhealthy fats low. A top-tier choice for your daily plate."
        ),
        "B": (
            "#85BB2F",
            "Light & Well-Balanced — Offers solid nutritional value with only minor trade-offs in salt or sugar. Fits easily into a clean, balanced routine."
        ),
        "C": (
            "#FECB02",
            "Enjoy Mindfully — Hits a middle ground between flavor and nutrition. Pair it with fresh greens or whole foods to round out your meal."
        ),
        "D": (
            "#EE8100",
            "Rich & Indulgent — Higher in fats, sugars, or sodium. Great when you're craving a treat, though best balanced with lighter meals throughout the day."
        ),
        "E": (
            "#E63E11",
            "Occasional Indulgence — Very dense in calories, sugars, or saturated fats with minimal protective nutrients. Best savored in small, occasional servings."
        )
    }

    bg_color, description = grade_info.get(grade, ("#333333", ""))
    # Dark green text on yellow Grade C for crisp contrast, white on all other grades
    text_color = "#1B4332" if grade == "C" else "#FFFFFF"

    st.markdown(
        f"""
        <div style="background-color:{bg_color}; padding:22px 24px; border-radius:12px; text-align:center; margin-top:20px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);">
            <h2 style="margin:0 0 8px 0; color:{text_color} !important;">Predicted Nutri-Score: Grade {grade}</h2>
            <p style="margin:0; font-size:15px; line-height:1.5; color:{text_color} !important;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# 5. Fixed bottom-left social links
st.markdown(
    """
    <div class="bottom-left-links">
        <a href="https://www.linkedin.com/in/ira-s/" target="_blank">LinkedIn</a>
        <a href="https://github.com/sinceayra" target="_blank">GitHub</a>
        <a href="https://www.instagram.com/sinceayra/" target="_blank">Instagram</a>
    </div>
    """,
    unsafe_allow_html=True
)
