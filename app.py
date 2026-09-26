"""
PakWheels Used Car Price Predictor
------------------------------------
A professional Streamlit GUI for the PakWheels used-car price prediction
model (SVR pipeline trained on PakWheels listings).

Author  : Abdul Rehman
GitHub  : https://github.com/datawithabdulrehman
LinkedIn: https://www.linkedin.com/in/datawithabdulrehman
Kaggle  : https://www.kaggle.com/datawithabxrehman

Run with:
    streamlit run app.py
"""

import datetime
import os

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# sklearn cross-version safety net
# The model was pickled with scikit-learn 1.6.1. If it is ever loaded
# with a newer/older scikit-learn that renamed an internal helper class,
# this shim keeps the file loadable instead of crashing on import.
# (Installing the exact scikit-learn version in requirements.txt avoids
# needing this, but the shim is a harmless safety net.)
# ----------------------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


# ----------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="PakWheels Price Predictor | Abdul Rehman",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "pakwheels_best_price_model.joblib")
METADATA_PATH = os.path.join(APP_DIR, "pakwheels_model_metadata.joblib")
CURRENT_YEAR = datetime.date.today().year

BRANDS = [
    "Aion", "Audi", "BAIC", "BMW", "BYD", "Bentley", "Changan", "Chery",
    "Chevrolet", "Chrysler", "DFSK", "Daewoo", "Daihatsu", "Deepal",
    "Dodge", "Dongfeng", "FAW", "Ford", "Forthing", "GAC", "GMC", "GUGO",
    "Haval", "Honda", "Honri", "Hyundai", "Inverex", "Isuzu", "JAC",
    "Jaecoo", "Jeep", "Jetour", "KIA", "Kaiyi", "Land Rover", "Lexus",
    "MG", "MINI", "Mazda", "Mercedes Benz", "Mitsubishi", "Nissan",
    "Omoda", "Peugeot", "Polaris", "Porsche", "Prince", "Proton",
    "Range Rover", "Rinco", "Rolls Royce", "Seres", "Smart", "SsangYong",
    "Subaru", "Suzuki", "Tank", "Tesla", "Toyota", "United", "Volkswagen",
]
FUEL_TYPES = ["petrol", "diesel", "hybrid", "electric", "cng", "phev", "reev"]
TRANSMISSIONS = ["automatic", "manual"]

AUTHOR = {
    "name": "Abdul Rehman",
    "github": "https://github.com/datawithabdulrehman",
    "linkedin": "https://www.linkedin.com/in/datawithabdulrehman",
    "kaggle": "https://www.kaggle.com/datawithabxrehman",
}

# ----------------------------------------------------------------------
# Theme — clean white base with a blue / teal automotive accent
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        :root {
            --primary: #1456F0;
            --primary-dark: #0B3FC4;
            --accent: #00C2A8;
            --bg-soft: #F6F8FC;
            --text-dark: #10182B;
            --text-muted: #5B6478;
            --border: #E4E8F1;
        }

        html, body, [class*="css"]  {
            font-family: "Segoe UI", Inter, "Helvetica Neue", Arial, sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #FBFCFE 0%, #F3F6FB 100%);
        }

        #MainMenu, footer {visibility: hidden;}

        /* Hero header */
        .hero {
            background: linear-gradient(120deg, var(--primary) 0%, var(--accent) 100%);
            border-radius: 18px;
            padding: 30px 36px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 10px 30px rgba(20, 86, 240, 0.18);
        }
        .hero h1 {
            font-size: 30px;
            font-weight: 800;
            margin: 0 0 4px 0;
            color: white;
        }
        .hero p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }
        .badge-row a {
            text-decoration: none;
            color: white;
            background: rgba(255,255,255,0.16);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            margin-right: 8px;
            display: inline-block;
            margin-top: 14px;
            border: 1px solid rgba(255,255,255,0.35);
        }
        .badge-row a:hover {
            background: rgba(255,255,255,0.30);
        }

        /* Cards */
        .card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 4px 16px rgba(16, 24, 43, 0.04);
        }

        .section-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-dark);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 14px;
        }

        /* Prediction result card */
        .result-card {
            background: linear-gradient(135deg, #0B3FC4 0%, #00C2A8 100%);
            border-radius: 18px;
            padding: 28px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 28px rgba(11, 63, 196, 0.22);
        }
        .result-card .label {
            font-size: 14px;
            opacity: 0.85;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .result-card .price {
            font-size: 42px;
            font-weight: 800;
            margin: 6px 0 2px 0;
        }
        .result-card .sub {
            font-size: 14px;
            opacity: 0.9;
        }

        /* Buttons */
        .stButton>button, .stFormSubmitButton>button {
            background: linear-gradient(120deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 700;
            width: 100%;
            box-shadow: 0 6px 16px rgba(20, 86, 240, 0.25);
        }
        .stButton>button:hover, .stFormSubmitButton>button:hover {
            background: linear-gradient(120deg, var(--primary-dark) 0%, var(--primary) 100%);
        }

        .footer-note {
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 30px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }
        .footer-note a {
            color: var(--primary);
            font-weight: 600;
            text-decoration: none;
            margin: 0 6px;
        }

        [data-testid="stMetricValue"] {
            color: var(--primary-dark);
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_metadata(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return {}


def format_pkr(amount_pkr: float) -> str:
    """Format a PKR amount with thousands separators."""
    return f"PKR {amount_pkr:,.0f}"


def format_lacs(amount_pkr: float) -> str:
    """Format a PKR amount in the local Lac/Crore convention."""
    lacs = amount_pkr / 100_000
    if lacs >= 100:
        crores = lacs / 100
        return f"{crores:,.2f} Crore"
    return f"{lacs:,.2f} Lac"


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🚗 PakWheels Used Car Price Predictor</h1>
        <p>Machine-learning powered estimate of a used car's fair market price in Pakistan,
        trained on real PakWheels listings.</p>
        <div class="badge-row">
            <a href="{AUTHOR['github']}" target="_blank">💻 GitHub</a>
            <a href="{AUTHOR['linkedin']}" target="_blank">🔗 LinkedIn</a>
            <a href="{AUTHOR['kaggle']}" target="_blank">📊 Kaggle</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load model + metadata (with friendly error handling)
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found: `{os.path.basename(MODEL_PATH)}`. "
        "Place it in the same folder as app.py and refresh the page."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "Could not load the model file. This is usually a scikit-learn "
        "version mismatch — make sure the exact version pinned in "
        "requirements.txt is installed.\n\n"
        f"Details: {e}"
    )
    st.stop()

metadata = load_metadata(METADATA_PATH)

# ----------------------------------------------------------------------
# Sidebar — model info
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Model Information")
    if metadata:
        st.metric("Algorithm", metadata.get("best_model", "N/A"))
        col_a, col_b = st.columns(2)
        col_a.metric("R² Score", f"{metadata.get('test_r2', 0):.3f}")
        col_b.metric("MAE", f"{metadata.get('test_mae_million_pkr', 0):.2f}M")
        st.metric("RMSE", f"{metadata.get('test_rmse_million_pkr', 0):.2f}M PKR")
        st.caption(
            "Metrics measured on a held-out test set. "
            "MAE/RMSE are in millions of PKR."
        )
    else:
        st.info("No metadata file found — showing predictions only.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This app wraps a trained scikit-learn pipeline "
        "(imputation → scaling/one-hot encoding → SVR regressor) "
        "built from PakWheels used-car listings. "
        "Predictions are estimates, not guarantees of sale price."
    )
    st.markdown("---")
    st.markdown(
        f"""
        **Built by {AUTHOR['name']}**

        [💻 GitHub]({AUTHOR['github']}) &nbsp;|&nbsp;
        [🔗 LinkedIn]({AUTHOR['linkedin']}) &nbsp;|&nbsp;
        [📊 Kaggle]({AUTHOR['kaggle']})
        """
    )

# ----------------------------------------------------------------------
# Main layout — input form (left) + result (right)
# ----------------------------------------------------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔧 Vehicle Details</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", options=BRANDS, index=BRANDS.index("Toyota"))
            year = st.number_input(
                "Manufacturing Year",
                min_value=1950,
                max_value=CURRENT_YEAR,
                value=2018,
                step=1,
            )
            fuel_type = st.selectbox(
                "Fuel Type",
                options=FUEL_TYPES,
                format_func=lambda x: x.upper(),
                index=FUEL_TYPES.index("petrol"),
            )
        with c2:
            transmission = st.selectbox(
                "Transmission",
                options=TRANSMISSIONS,
                format_func=lambda x: x.capitalize(),
            )
            engine_cc = st.number_input(
                "Engine (CC)", min_value=50, max_value=8000, value=1300, step=50
            )
            mileage_km = st.number_input(
                "Mileage (KM)", min_value=0, max_value=1_000_000, value=60_000, step=1000
            )

        vehicle_age = CURRENT_YEAR - int(year)
        st.caption(f"Computed vehicle age: **{vehicle_age} year(s)** (as of {CURRENT_YEAR})")

        submitted = st.form_submit_button("🔮 Predict Price")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card" style="min-height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Predicted Price</div>', unsafe_allow_html=True)

    if "submitted_once" not in st.session_state:
        st.session_state.submitted_once = False

    if submitted:
        st.session_state.submitted_once = True

        input_df = pd.DataFrame([{
            "brand": brand,
            "year": int(year),
            "fuel_type": fuel_type,
            "transmission": transmission,
            "engine_cc": float(engine_cc),
            "mileage_km": float(mileage_km),
            "vehicle_age": vehicle_age,
        }])

        try:
            predicted_million = float(model.predict(input_df)[0])
            predicted_million = max(predicted_million, 0.0)
            predicted_pkr = predicted_million * 1_000_000

            result_html = (
                '<div class="result-card">'
                f'<div style="font-size:18px; font-weight:800;">{brand} &nbsp;·&nbsp; {int(year)}</div>'
                '<div class="label" style="margin-top:10px;">Estimated Market Price</div>'
                f'<div class="price">{format_pkr(predicted_pkr)}</div>'
                f'<div class="sub">≈ {format_lacs(predicted_pkr)} &nbsp;•&nbsp; {predicted_million:.2f} Million PKR</div>'
                '</div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)

            st.write("")
            st.markdown("**Input Summary**")
            summary_rows = {
                "Feature": ["Brand", "Year", "Fuel Type", "Transmission",
                            "Engine (CC)", "Mileage (KM)", "Vehicle Age"],
                "Value": [brand, int(year), fuel_type.upper(), transmission.capitalize(),
                          f"{int(engine_cc):,}", f"{int(mileage_km):,}", f"{vehicle_age} yrs"],
            }
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            if metadata.get("test_mae_million_pkr"):
                mae = metadata["test_mae_million_pkr"]
                st.caption(
                    f"Typical model error on unseen data: ± {mae:.2f} Million PKR (MAE). "
                    "Treat this as an estimate — actual listing prices vary with condition, "
                    "location, and market demand."
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    elif not st.session_state.submitted_once:
        st.info("Fill in the vehicle details on the left and click **Predict Price** to get an estimate.")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer-note">
        Built with ❤️ by <strong>{AUTHOR['name']}</strong> &nbsp;·&nbsp;
        <a href="{AUTHOR['github']}" target="_blank">GitHub</a>
        <a href="{AUTHOR['linkedin']}" target="_blank">LinkedIn</a>
        <a href="{AUTHOR['kaggle']}" target="_blank">Kaggle</a>
        &nbsp;·&nbsp; Model: SVR pipeline trained on PakWheels listings
    </div>
    """,
    unsafe_allow_html=True,
)