from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved KNN model
MODEL_PATH = os.path.join('Knn_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Define the expected feature names (in correct order)
feature_names = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1',
    'cp_0', 'cp_1', 'cp_2', 'cp_3', 'fbs_0', 'fbs_1', 'restecg_0', 'restecg_1',
    'restecg_2', 'exang_0', 'exang_1', 'slope_0', 'slope_1', 'slope_2', 'ca_0',
    'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_0', 'thal_1', 'thal_2', 'thal_3'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect input features safely
            features = [float(request.form.get(feature, 0)) for feature in feature_names]
            features_np = np.array(features).reshape(1, -1)

            # Make prediction
            raw_prediction = model.predict(features_np)[0]
            if raw_prediction == 1:
                prediction = "Yes, you have heart disease."
            else:
                prediction = "No, you do not have heart disease."
        except Exception as e:
            prediction = f"Prediction error: {str(e)}"

    return render_template('index.html', prediction=prediction, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)