from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
with open('lm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input values from the form
            avg_income = float(request.form['avg_income'])
            avg_age = float(request.form['avg_age'])
            avg_rooms = float(request.form['avg_rooms'])
            avg_bedrooms = float(request.form['avg_bedrooms'])
            area_population = float(request.form['area_population'])

            # Prepare the data for prediction
            features = np.array([[avg_income, avg_age, avg_rooms, avg_bedrooms, area_population]])

            # Apply the scaler
            features_scaled = scaler.transform(features)

            # Make the prediction
            prediction = model.predict(features_scaled)

            return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')

        except Exception as e:
            return render_template('index.html', error_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
