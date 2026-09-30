import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Rwanda House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.sav")


try:
    model = load_model()
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.info(
        "Make sure house_price_model.sav is in the same folder as app.py "
        "and that the scikit-learn version in requirements.txt matches the model."
    )
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Rwanda House Price Prediction System")

st.write(
    "Enter the characteristics of a house to estimate its selling price "
    "in million Rwandan Francs (RWF)."
)

st.divider()


# ============================================================
# INPUTS
# ============================================================

st.subheader("House Information")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (m²)",
        min_value=1.0,
        max_value=1000.0,
        value=150.0,
        step=1.0
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    house_age = st.number_input(
        "House Age (Years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

with col2:
    distance = st.number_input(
        "Distance to City Centre (km)",
        min_value=0.0,
        max_value=100.0,
        value=4.0,
        step=0.1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=2,
        step=1
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "Gasabo",
            "Huye",
            "Kicukiro",
            "Kigali City",
            "Musanze",
            "Nyarugenge"
        ]
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict House Price",
    type="primary",
    use_container_width=True
):

    if area <= 0:
        st.error("Area must be greater than zero.")
        st.stop()

    new_house = pd.DataFrame({
        "Area_m2": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "House_Age_Years": [house_age],
        "Distance_to_City_km": [distance],
        "Parking_Spaces": [parking],
        "Neighborhood": [neighborhood]
    })

    try:
        prediction = float(model.predict(new_house)[0])

        st.success("Prediction completed successfully!")

        st.metric(
            "Estimated House Price",
            f"{prediction:,.2f} Million RWF"
        )

        st.subheader("House Information")

        summary = pd.DataFrame({
            "Feature": [
                "Area",
                "Bedrooms",
                "Bathrooms",
                "House Age",
                "Distance to City Centre",
                "Parking Spaces",
                "Neighborhood"
            ],
            "Value": [
                f"{area:.1f} m²",
                str(bedrooms),
                str(bathrooms),
                f"{house_age:.1f} years",
                f"{distance:.1f} km",
                str(parking),
                neighborhood
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Price Visualization")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(["Predicted Price"], [prediction])
        ax.set_ylabel("Million RWF")
        ax.set_title("Estimated House Price")
        st.pyplot(fig)

        report = pd.DataFrame({
            "Area_m2": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "House_Age_Years": [house_age],
            "Distance_to_City_km": [distance],
            "Parking_Spaces": [parking],
            "Neighborhood": [neighborhood],
            "Predicted_Price_Million_RWF": [prediction]
        })

        st.download_button(
            "📥 Download Prediction Report",
            data=report.to_csv(index=False),
            file_name="house_price_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.exception(e)


st.divider()

st.caption(
    "Rwanda House Price Prediction System | Multiple Linear Regression"
)
