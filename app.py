from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('dl_housing.keras')

scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [float(data[key]) for key in data.keys()]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    output = prediction[0][0]
    
    return render_template('home.html', prediction=f'Predicted House Price: ${output:.2f}')

@app.route('/aboutProject')
def aboutProject():
    return render_template('aboutProject.html')

@app.route('/aboutMe')
def aboutMe():
    return render_template('aboutMe.html')

if __name__ == '__main__':
    app.run(debug=True)
