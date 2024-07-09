from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd

from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/ivan.osipchyk.work/end-to-end-ml-project.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"]="your_username"
# os.environ["MLFLOW_TRACKING_PASSWORD"]="your_password"


@app.route('/', methods=['GET'])
def home_page():
    return render_template("index.html")


@app.route('/train', methods=['GET'])
def train():
    os.system("python3 main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            data = [fixed_acidity, volatile_acidity, citric_acid,
                    residual_sugar, chlorides, free_sulfur_dioxide,
                    total_sulfur_dioxide, density, pH,
                    sulphates, alcohol]

            data = np.array(data).reshape(1, 11)

            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)[0]

            prediction_mapping = {
                0: 'Bad',
                1: 'Fair',
                2: 'Good'
            }

            quality = prediction_mapping[prediction]

            return render_template('results.html', prediction=quality)
        except Exception as e:
            print('The Exception message is: ', e)
            return render_template('error.html', error_message=str(e))

    else:
        render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
