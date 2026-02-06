import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq
import sklearn
import xgboost

# --- Page Config ---
st.set_page_config(page_title="Real Estate AI Expert", page_icon="🏠", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box_shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants & Config ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = "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" # Hardcoded as requested
MODEL_FILES = {
    "classifier": "category_classifier.pkl",
    "regressor": "price_regressor.pkl",
    "stats": "market_stats.pkl",
    "scaler": "scaler.pkl"
}

# --- Load Assets ---
@st.cache_resource
def load_assets():
    assets = {}
    try:
        assets["clf"] = joblib.load(os.path.join(BASE_PATH, MODEL_FILES["classifier"]))
        assets["reg"] = joblib.load(os.path.join(BASE_PATH, MODEL_FILES["regressor"]))
        assets["stats"] = joblib.load(os.path.join(BASE_PATH, MODEL_FILES["stats"]))
        # Try to load scaler, if not found, handle gracefully (though we created it)
        scaler_path = os.path.join(BASE_PATH, MODEL_FILES["scaler"])
        if os.path.exists(scaler_path):
            assets["scaler"] = joblib.load(scaler_path)
        else:
            assets["scaler"] = None
            st.warning("Scaler not found. Predictions might be inaccurate.")
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None
    return assets

assets = load_assets()

# --- Logic ---
def get_city_enc(city, city_map):
    # Retrieve city encoding from stats, default to median of map if unknown
    return city_map.get(city, np.median(list(city_map.values())))

def predict(inputs, assets):
    # Prepare DataFrame
    # Model expects specific feature order as defined in scaler training
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 
                'yr_built', 'yr_renovated', 'city_enc', 'house_age', 'was_renovated']
    
    # Impute missing inputs based on heuristics
    data = {
        'bedrooms': inputs['bedrooms'],
        'bathrooms': inputs['bathrooms'],
        'sqft_living': inputs['sqft_living'],
        'sqft_lot': inputs['sqft_living'] * 1.5, # Assumed
        'floors': inputs['floors'],
        'waterfront': 1 if inputs['waterfront'] else 0,
        'view': inputs['view'],
        'condition': inputs['condition'],
        'sqft_above': inputs['sqft_living'], # Simplify: assume mostly above ground
        'sqft_basement': 0, # Simplify
        'yr_built': 2026 - inputs['house_age'],
        'yr_renovated': 2020 if inputs['was_renovated'] else 0, # Placeholder year if renovated
        'city_enc': get_city_enc(inputs['city'], assets['stats']['city_map']),
        'house_age': inputs['house_age'],
        'was_renovated': 1 if inputs['was_renovated'] else 0
    }
    
    df = pd.DataFrame([data])
    df = df[features] # Ensure order
    
    # Scale
    if assets['scaler']:
        X_scaled = assets['scaler'].transform(df)
        X_final = pd.DataFrame(X_scaled, columns=features)
    else:
        X_final = df

    # Predict
    cat_pred = assets['clf'].predict(X_final)[0]
    log_price = assets['reg'].predict(X_final)[0]
    price_pred = np.expm1(log_price) # Inverse log
    
    return cat_pred, price_pred

def generate_explanation(inputs, price, category, assets):
    client = Groq(api_key=GROQ_API_KEY)
    
    market_median = assets['stats']['median_price']
    city_stats = assets['stats']['city_map']
    city_val = city_stats.get(inputs['city'], "Unknown")
    
    prompt = f"""
    You are a Real Estate Expert. Analyze this property:
    - Details: {inputs['bedrooms']} bed, {inputs['bathrooms']} bath, {inputs['sqft_living']} sqft in {inputs['city']}.
    - Condition: {inputs['condition']}/5, View: {inputs['view']}/4.
    - Predicted Price: ${price:,.0f}
    - Category: {category}
    
    Market Context:
    - Overall Market Median: ${market_median:,.0f}
    - City Value Index: {city_val}
    
    Explain why this price is justified based on the features and market. 
    Is it a fair market value? Keep it concise (3-4 sentences).
    """
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate explanation: {e}"

# --- UI Layout ---
st.title("🏠 AI Real Estate Valuation Expert")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Property Details")
    with st.form("input_form"):
        city = st.selectbox("City", options=sorted(list(assets['stats']['city_map'].keys())))
        bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.5)
        sqft_living = st.number_input("Sqft Living", min_value=100, value=2000, step=50)
        floors = st.number_input("Floors", min_value=1.0, value=1.0, step=0.5)
        house_age = st.number_input("House Age (Years)", min_value=0, value=10)
        
        c1, c2 = st.columns(2)
        with c1:
            condition = st.slider("Condition (1-5)", 1, 5, 3)
            waterfront = st.checkbox("Waterfront")
        with c2:
            view = st.slider("View (0-4)", 0, 4, 0)
            was_renovated = st.checkbox("Renovated")
            
        submitted = st.form_submit_button("Predict Value")

with col2:
    if submitted and assets:
        inputs = {
            "city": city, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "sqft_living": sqft_living, "floors": floors, "house_age": house_age,
            "condition": condition, "view": view, "waterfront": waterfront,
            "was_renovated": was_renovated
        }
        
        with st.spinner("Analyzing Market Data..."):
            cat, price = predict(inputs, assets)
            explanation = generate_explanation(inputs, price, cat, assets)
        
        st.subheader("Valuation Report")
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Estimated Value</h3>
                <h2 style="color: #28a745;">${price:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Property Category</h3>
                <h2 style="color: #007bff;">{cat}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### 🤖 AI Market Analysis")
        st.info(explanation)

    elif not submitted:
        st.info("Enter property details and click Predict to see the valuation.")
