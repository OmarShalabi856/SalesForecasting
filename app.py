from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

product_mapping = pd.read_csv(r'./product_mapping.csv')
model = load_model(r'./best_model.h5')
with open(r"./Scaker.pkl",'rb') as f:
    scaler = pickle.load(f)

@app.route('/SalesForecast', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        price = float(request.form['price'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        product_name = request.form['product_name']
        product_name = "_" + product_name
        product_index = product_mapping.loc[product_mapping['product'] == product_name, 'product_index'].iloc[0]
        product_vector = np.zeros(len(product_mapping))
        product_vector[product_index] = 1
        features = [price] + product_vector.tolist() + [year, day, month]
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        y_pred = model.predict(price, product_name, year, day,month)
        response = {'prediction': y_pred}
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
