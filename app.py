import streamlit as st
import pandas as pd
import joblib


# -------------------------
# PAGE CONFIGURATION
# -------------------------

st.set_page_config(
    page_title="California House Value Predictor",
    page_icon="🏠",
    layout="wide"
)


# -------------------------
# LOAD TRAINED MODEL
# -------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/random_forest_model.pkl")


model = load_model()


# -------------------------
# HEADER
# -------------------------

st.title("🏠 California House Value Predictor")

st.write(
    "Estimate the median house value of a California census block "
    "using a trained Random Forest machine learning model."
)

st.divider()


# -------------------------
# MODEL INFORMATION
# -------------------------

st.subheader("Model Performance")

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric(
        label="Model",
        value="Random Forest"
    )

with metric_col2:
    st.metric(
        label="Test R²",
        value="0.806"
    )

with metric_col3:
    st.metric(
        label="Mean Absolute Error",
        value="~$32,690"
    )

st.caption(
    "The model was evaluated on 4,128 unseen test samples from the "
    "California Housing dataset."
)

st.divider()


# -------------------------
# USER INPUTS
# -------------------------

st.subheader("Enter Housing Information")

left_column, right_column = st.columns(2)


# LEFT COLUMN
with left_column:

    st.markdown("#### Property & Income")

    med_inc = st.number_input(
        "Median Income",
        min_value=0.5,
        max_value=15.0,
        value=5.0,
        step=0.1,
        help="Median income for the census block, measured in tens of thousands of dollars."
    )

    house_age = st.number_input(
        "Median House Age",
        min_value=1.0,
        max_value=52.0,
        value=25.0,
        step=1.0,
        help="Median age of houses within the census block."
    )

    ave_rooms = st.number_input(
        "Average Rooms per Household",
        min_value=0.8,
        max_value=142.0,
        value=6.0,
        step=0.1
    )

    ave_bedrooms = st.number_input(
        "Average Bedrooms per Household",
        min_value=0.3,
        max_value=35.0,
        value=1.0,
        step=0.1
    )


# RIGHT COLUMN
with right_column:

    st.markdown("#### Location & Population")

    population = st.number_input(
        "Block Population",
        min_value=3.0,
        max_value=35682.0,
        value=1200.0,
        step=100.0,
        help="Total population of the census block."
    )

    ave_occupancy = st.number_input(
        "Average Household Occupancy",
        min_value=0.6,
        max_value=1244.0,
        value=3.0,
        step=0.1,
        help="Average number of people living in each household."
    )

    latitude = st.number_input(
        "Latitude",
        min_value=32.54,
        max_value=41.95,
        value=34.05,
        step=0.01,
        help="Geographical latitude of the census block."
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-124.35,
        max_value=-114.31,
        value=-118.25,
        step=0.01,
        help="Geographical longitude of the census block."
    )


# -------------------------
# PREDICTION
# -------------------------

st.divider()

st.subheader("House Value Prediction")

if st.button(
    "Estimate House Value",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrooms],
        "Population": [population],
        "AveOccup": [ave_occupancy],
        "Latitude": [latitude],
        "Longitude": [longitude]
    })

    prediction = model.predict(input_data)

    predicted_value = prediction[0] * 100000

    st.success("Prediction complete")

    st.metric(
        label="Estimated Median House Value",
        value=f"${predicted_value:,.0f}"
    )


# -------------------------
# ABOUT THE MODEL
# -------------------------

st.divider()

with st.expander("How does this model work?"):

    st.write(
        """
        This application uses a Random Forest regression model trained on
        the California Housing dataset.

        The model receives eight features describing a California census
        block, including median income, house age, average rooms,
        population and geographical location.

        During development, Linear Regression was used as a baseline model.
        The Random Forest achieved substantially better performance and was
        selected as the final model.

        The model was also evaluated using cross-validation and
        hyperparameter tuning before being saved and connected to this
        Streamlit application.
        """
    )


# -------------------------
# DISCLAIMER
# -------------------------

st.info(
    "Educational project: predictions are based on historical California "
    "housing data and should not be used as real-world property valuations."
)