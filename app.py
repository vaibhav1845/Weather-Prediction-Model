from flask import Flask, request, render_template
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

# Load your trained KNN model
model = joblib.load('knn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        
        # Prepare the input for prediction
        input_data = np.array([[precipitation, temp_max, temp_min, wind]])
        
        # Predict weather
        prediction = model.predict(input_data)[0]
        
        # Map the prediction to a readable format
        weather_map = {0: 'Clear', 1: 'Cloudy', 2: 'Rainy', 3: 'Snowy', 4: 'Windy'}
        weather = weather_map.get(prediction, 'Unknown')
        
        return render_template('result.html', weather=weather)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True) 
