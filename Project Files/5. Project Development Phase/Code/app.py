import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.pkl'
DATA_PATH = 'Crop_recommendation.csv'

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please run train_model.py first.")

# Load the dataset for dashboard analytics
df_crops = None
crop_averages = {}
crop_list = []
if os.path.exists(DATA_PATH):
    try:
        df_crops = pd.read_csv(DATA_PATH)
        # Rename columns to standard lowercase long names
        rename_dict = {'N': 'nitrogen', 'P': 'phosphorous', 'K': 'potassium'}
        df_crops.rename(columns=lambda x: rename_dict.get(x, x), inplace=True)
        
        # Calculate averages per crop for suitability comparisons
        grouped = df_crops.groupby('label').mean()
        crop_averages = grouped.to_dict(orient='index')
        crop_list = sorted(df_crops['label'].unique().tolist())
        print("Dataset loaded and crop averages calculated.")
    except Exception as e:
        print(f"Error loading dataset: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html', prediction_text=None, crop_list=crop_list)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('findyourcrop.html', 
                               prediction_text="Error: Model not loaded.", 
                               crop_list=crop_list)
    try:
        # Extract features by explicit names to ensure correct order
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        features = pd.DataFrame([[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]],
                                columns=['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Get class probabilities if supported, or just prediction
        prediction = model.predict(features)[0]
        
        # Generate some additional suitability advice based on averages
        suitability_msg = ""
        if prediction in crop_averages:
            avg = crop_averages[prediction]
            issues = []
            if rainfall < avg['rainfall'] * 0.7:
                issues.append("lower rainfall than optimal (suggest supplemental irrigation)")
            elif rainfall > avg['rainfall'] * 1.3:
                issues.append("higher rainfall than optimal (ensure proper drainage)")
            if ph < avg['ph'] - 1.0:
                issues.append("high soil acidity for this crop (consider applying lime)")
            elif ph > avg['ph'] + 1.0:
                issues.append("high soil alkalinity for this crop (consider applying organic matter/sulfur)")
            
            if issues:
                suitability_msg = "Note: Your inputs show " + ", and ".join(issues) + "."
            else:
                suitability_msg = "Your soil and environmental conditions are highly compatible with this crop!"

        prediction_text = f"Recommended Crop: {prediction.capitalize()}"
        return render_template('findyourcrop.html', 
                               prediction_text=prediction_text, 
                               predicted_crop=prediction,
                               suitability_msg=suitability_msg,
                               crop_list=crop_list,
                               # Send inputs back to form
                               inputs={
                                   'nitrogen': nitrogen,
                                   'phosphorous': phosphorous,
                                   'potassium': potassium,
                                   'temperature': temperature,
                                   'humidity': humidity,
                                   'ph': ph,
                                   'rainfall': rainfall
                               })
    except Exception as e:
        return render_template('findyourcrop.html', 
                               prediction_text=f"Error in prediction: {str(e)}", 
                               crop_list=crop_list)

@app.route('/dashboard')
def dashboard():
    # Render dashboard template and pass crop lists and metrics as JSON
    return render_template('dashboard.html', crop_list=crop_list)

@app.route('/api/crop-averages')
def api_crop_averages():
    # Return crop averages for client-side Chart.js visualizations
    return jsonify(crop_averages)

@app.route('/api/crop-data')
def api_crop_data():
    if df_crops is None:
        return jsonify([])
    # Limit response size by sampling or sending necessary columns only
    # Sample 500 rows for bivariate scatter plots to keep frontend fast
    sample_df = df_crops.sample(n=min(500, len(df_crops)), random_state=42)
    return jsonify(sample_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
