from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models
model = joblib.load("models/aviation_risk_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Create empty feature vector
    features = np.zeros(len(feature_names))

    # Fill features
    for i, f in enumerate(feature_names):
        if f in data:
            features[i] = data[f]

    # Scale if Logistic Regression
    if "LogisticRegression" in str(type(model)):
        features = scaler.transform([features])
    else:
        features = [features]

    # Predict
    prob = model.predict_proba(features)[0, 1]
    risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

    return jsonify({
        "probability": float(prob),
        "risk": risk
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

from flask_cors import CORS
CORS(app)

