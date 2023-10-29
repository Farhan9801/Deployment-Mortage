import pickle
from flask import Flask, render_template, request, jsonify
import pandas as pd
from myProject1.preprocessing_regression import data_processing_pipeline_regression
from myProject1.FeatureEngineering import feature_engineering_pipeline

from myProject1.feature_addition import create_features


flask_app = Flask(__name__)

log_model = pickle.load(open('LogisticRegression.pkl', 'rb'))
Reg_model = pickle.load(open('RegressionModel.pkl', 'rb'))


@flask_app.route("/")
def home():
    return render_template("base.html")


@flask_app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        print(df)
        prediction = log_model.predict(df)
        print('PREDICTION', prediction)
        prediction_list = prediction.tolist()  # Convert NumPy array to list
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    flask_app.run(debug=True)