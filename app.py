import streamlit as st
import pandas as pd
import numpy as np
import joblib

df = pd.read_excel('realistic_housing_data.xlsx')
pipeline = joblib.load("house_price_predictor.pkl")

st.set_page_config(page_title="House Price Predictor")

st.markdown("""
    <h1 style="text-align:center; color:#4CAF50; font-size:42px;">
        🏡 House Price Predictor
    </h1>
    <p style="text-align:center; font-size:18px;">
        Enter house details below to estimate the price instantly 📊
    </p>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1])

# LEFT SECTION
with left:
    st.subheader("🏠 Property Details")
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    sqft = st.number_input("Square Feet (sqft)", 500, 4000, 2000)
    lot_size = st.number_input("Lot Size (sqft)", 1000, 20000, 8000)
    age = st.slider("Age of House (years)", 0, 50, 10)
    year_built = st.number_input("Year Built", 1970, 2024, 2005)
    house_type = st.selectbox("House Type", sorted(df['house_type'].unique()))

# RIGHT SECTION
with right:
    st.subheader("🏗 Additional Features")
    garage = st.slider("Garage Capacity", 0, 3, 1)
    condition = st.slider("Condition (1 = Poor, 5 = Excellent)", 1, 5, 3)
    has_pool = st.selectbox("Has Pool?", [0, 1])
    has_fireplace = st.selectbox("Has Fireplace?", [0, 1])
    has_basement = st.selectbox("Has Basement?", [0, 1])
    school_rating = st.slider("School Rating", 1, 10, 6)
    location = st.selectbox("Location", sorted(df['location'].unique()))


# Predict Button
center = st.columns([3, 2, 3])
with center[1]:
    submitted = st.button("🔍 Predict Price", use_container_width=True)

# ------------------------ PREDICTION SECTION ------------------------
if submitted:
    input_data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft": sqft,
        "lot_size": lot_size,
        "age": age,
        "year_built": year_built,
        "garage": garage,
        "condition": condition,
        "has_pool": has_pool,
        "has_fireplace": has_fireplace,
        "has_basement": has_basement,
        "school_rating": school_rating,
        "location": location,
        "house_type": house_type
    }

    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]

    

    # Price Value
    st.markdown(
        f"""
        <h2 style="text-align:center; color:#2E7D32;">
            ₹ {prediction:,.2f}
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Category Message
    if prediction > 800000:
        st.info("🌟 Premium Property – Excellent Investment Potential!")
    elif prediction > 300000:
        st.info("⚡ Mid-range Property – Good Value for Money!")
    else:
        st.info("📉 Low-range Property – Affordable Segment.")

# Footer
st.markdown("---")
st.caption("Developed by Atchaya • 🔥 Random Forest Regression Model")
