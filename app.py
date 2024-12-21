import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json



app = Flask(__name__)

model = pickle.load(open('reg_model.pkl','rb'))
scaler = pickle.load(open('std_scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    new_data = np.array(list(data.values( ))).reshape(1,-1)
    print(new_data)
    transformed_data = scaler.transform(new_data)
    out = model.predict(transformed_data)
    print(out[0])
    return jsonify(out[0])


@app.route('/predict', methods = ['POST'])
def predict():
    # data = request.json['data']
    data = [float(x) for x in request.form.values()]

    new_data = np.array(data).reshape(1,-1)
    print(new_data)
    transformed_data = scaler.transform(new_data)
    out = model.predict(transformed_data)
    print(out[0])
    return render_template('home.html', prediction_text = f'The house price prediction is {out[0]}.')



if __name__=='__main__':
    app.run(debug = True)
