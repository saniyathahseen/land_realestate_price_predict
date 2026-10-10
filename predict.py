import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("land_prices.csv")


df["price_per_cent"] = df["price_lakhs"] / df["land_area_cents"]
df_processed = pd.get_dummies(
    df, columns=["location_name", "taluk", "village", "land_type"], drop_first=True
)
numeric_features = [
    "land_area_cents",
    "distance_to_school_km",
    "distance_to_airport_km",
    "distance_to_railway_station_km",
    "distance_to_hospital_km",
    "distance_to_medical_college_km",
    "distance_to_bus_stop_km",
    "distance_to_market_km",
]
dummy_features = [
    col
    for col in df_processed.columns
    if col.startswith("location_name_")
    or col.startswith("taluk_")
    or col.startswith("village_")
    or col.startswith("land_type_")
]
features = numeric_features + dummy_features

y = df["price_per_cent"]
x = df_processed[features]
# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# create a model to train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluate the model
df_test = pd.DataFrame({"actual": y_test, "predicted": y_pred})
# try new land data
new_land_data = {
    "land_area_cents": [10],
    "distance_to_school_km": [2],
    "distance_to_airport_km": [15],
    "distance_to_railway_station_km": [5],
    "distance_to_hospital_km": [3],
    "distance_to_medical_college_km": [4],
    "distance_to_bus_stop_km": [1],
    "distance_to_market_km": [2],
    "location_name_ABC": [1],
    "taluk_XYZ": [1],
    "village_DEF": [1],
    "land_type_residential": [1],
}
new_land_df = pd.DataFrame(new_land_data)
# ensure all dummy columns are present in the new data
for col in dummy_features:
    if col not in new_land_df.columns:
        new_land_df[col] = [0]


new_land_df = new_land_df[features]  # reorder columns to match training data
predicted_price_per_cent = model.predict(new_land_df)
print(f"Predicted price per cent for the new land data: {predicted_price_per_cent[0]:.2f} lakhs")


