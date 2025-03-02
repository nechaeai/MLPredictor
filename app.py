from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib

model, scaler = joblib.load("SEPModel.pkl")

features = ["Age", "Gender", "BMI", "BodyFat", "Myelopathy"]

app = Flask(__name__)
CORS(app)  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        age = float(data["age"])
        gender = 1 if data["gender"].lower() == "male" else 0
        bmi = float(data["bmi"])
        bodyfat = float(data["bodyfat"])
        myelopathy = int(data["myelopathy"])

        sample_df = pd.DataFrame([[age, gender, bmi, bodyfat, myelopathy]], columns=features)
        sample_scaled = scaler.transform(sample_df)
        prediction = model.predict(sample_scaled)[0]

        return jsonify({"angle": prediction[0], "amperage": prediction[1]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
