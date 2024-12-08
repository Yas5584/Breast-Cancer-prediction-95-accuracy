from flask import Flask, request, render_template, jsonify
import pandas as pd
import pickle
import numpy as np


application = Flask(__name__)
app = application

# Load the pre-trained SVM model and scaler
svc_classifier = pickle.load(open(r'C:\Users\ys136\Desktop\Data Science\End to End ML Projects\diabetes prediction using SVM\models\svc.pkl', 'rb'))  # Replace with your model's path
 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect input features from the form
        radius_mean = float(request.form.get('radius_mean'))
        texture_mean = float(request.form.get('texture_mean'))
        perimeter_mean = float(request.form.get('perimeter_mean'))
        area_mean = float(request.form.get('area_mean'))
        smoothness_mean = float(request.form.get('smoothness_mean'))
        compactness_mean = float(request.form.get('compactness_mean'))
        concavity_mean = float(request.form.get('concavity_mean'))
        concave_points_mean = float(request.form.get('concave_points_mean'))
        symmetry_mean = float(request.form.get('symmetry_mean'))
        fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))

        # Combine inputs into a single array
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]])



        # Predict using the loaded model
        result = svc_classifier.predict(input_data)

        # Convert the result to a readable format
        tumor_type = "Malignant" if result[0] == 1 else "Benign"

        return render_template('home.html', result=tumor_type)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5501, debug=True)
