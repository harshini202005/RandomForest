from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'f{i}']) for i in range(1, 5)]
        prediction = model.predict([features])[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
    except:
        result = "Invalid input. Please enter numbers only."

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
