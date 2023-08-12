# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# load the model and the scaler
model = pickle.load(open('/home/chenjieling/mysite/model.pkl', 'rb'))
scaler = pickle.load(open('/home/chenjieling/mysite/scaler.pkl', 'rb'))

@app.route('/' , methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        #get the data from the POST request
        input_data = request.form.to_dict()
        input_data = pd.DataFrame([input_data])
        input_dummy = pd.get_dummies(input_data, drop_first=True)

        # ensure columns match with what the scaler and model expect
        expected_columns = list(scaler.mean_)
        for column in expected_columns:
            if column not in input_dummy.columns:
                input_dummy[column] = 0

        # ensure order of columns matches original order
        input_dummy = input_dummy[expected_columns]

        standardized_data = scaler.transform(input_dummy.values)

        # make prediction using the model
        prediction = model.predict(standardized_data)
        probabilities = model.predict_proba(standardized_data)
        probability = probabilities[0][prediction[0]] * 100
        result = "Subscribed" if prediction[0] == 1 else "Not Subscribed"


        return render_template('bank_marketing_prediction.html', prediction_text='Predicted subscription: {} with confidence: {}'.format(result, probability))

    return render_template('bank_marketing_prediction.html')
