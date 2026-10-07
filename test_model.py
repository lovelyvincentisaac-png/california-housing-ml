import joblib
import pandas as pd

# -------------------------
# LOAD THE TRAINED MODEL
# -------------------------

model = joblib.load("model/random_forest_model.pkl")

print("Model loaded successfully!")


# -------------------------
# CREATE EXAMPLE INPUT
# -------------------------

example_house = pd.DataFrame({
    "MedInc": [5.0],
    "HouseAge": [25.0],
    "AveRooms": [6.0],
    "AveBedrms": [1.0],
    "Population": [1200.0],
    "AveOccup": [3.0],
    "Latitude": [34.05],
    "Longitude": [-118.25]
})


# -------------------------
# MAKE A PREDICTION
# -------------------------

prediction = model.predict(example_house)

predicted_value = prediction[0] * 100000

print(f"Predicted median house value: ${predicted_value:,.2f}")