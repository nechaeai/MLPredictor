import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

data_path = "updated_patient_data.csv"
df = pd.read_csv(data_path)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})  

features = ["Age", "Gender", "BMI", "BodyFat", "Myelopathy"]
targets = ["Angle", "Amperage"]

X = df[features]
y = df[targets]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Metrics:")
print(f"  Mean Absolute Error: {mae}")
print(f"  Mean Squared Error: {mse}")
print(f"  R-Squared: {r2}")

model_path = "SEPModel.pkl"
joblib.dump((model, scaler), model_path)
print(f"Model saved to: {model_path}")

def predict_angle_amperage(age, gender, bmi, bodyfat, myelopathy):
    gender = 1 if gender.lower() == "male" else 0  
    sample_df = pd.DataFrame([[age, gender, bmi, bodyfat, myelopathy]], columns=features)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    return prediction[0]

if __name__ == "__main__":
    print("Enter patient details to predict Angle and Amperage:")
    age = float(input("Age: "))
    gender = input("Gender (Male/Female): ")
    bmi = float(input("BMI: "))
    bodyfat = float(input("BodyFat: "))
    myelopathy = float(input("Myelopathy (0 or 1): "))
    
    predicted_values = predict_angle_amperage(age, gender, bmi, bodyfat, myelopathy)
    print(f"Predicted Angle and Amperage: {predicted_values}")
