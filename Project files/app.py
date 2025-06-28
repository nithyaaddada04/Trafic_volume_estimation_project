import numpy as np
import pickle
import os
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model, scaler and training columns
model = pickle.load(open("C:/Users/addad/OneDrive/Desktop/smart bridge/Flask/model.pkl", "rb"))
scaler = pickle.load(open("C:/Users/addad/OneDrive/Desktop/smart bridge/Flask/scaler (1).pkl", "rb"))
columns = pickle.load(open("C:/Users/addad/OneDrive/Desktop/smart bridge/Flask/columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST", "GET"])
def predict():
    # Collect form values
    raw_inputs = {
        'holiday': request.form.get('holiday'),
        'temp': float(request.form.get('temp')),
        'rain': float(request.form.get('rain')),
        'snow': float(request.form.get('snow')),
        'weather': request.form.get('weather'),
        'year': int(request.form.get('year')),
        'month': int(request.form.get('month')),
        'day': int(request.form.get('day')),
        'hours': int(request.form.get('hours')),
        'minutes': int(request.form.get('minutes')),
        'seconds': int(request.form.get('seconds'))
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_inputs])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Return result
    text = "Estimated Traffic Volume is "
    return render_template("result.html", prediction_text=text + str(int(prediction[0])))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, debug=True, use_reloader=False)
