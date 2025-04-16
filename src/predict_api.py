import os
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Paths to the processed data and the model
MODEL_PATH = '/home/bhullar295/Edmonton_Food_Drive_Project/models/random_forest_regressor_model.pkl'
PROCESSED_DATA_PATH = '/home/bhullar295/Edmonton_Food_Drive_Project/data/processed/Cleaned_food_drive_data.csv'

# Load model and label encoder
model = load(MODEL_PATH)
data = pd.read_csv(PROCESSED_DATA_PATH)

# Load label encoder for 'Completed More Than One Route' and 'Comment Sentiments'
le = LabelEncoder()

# Clean column names (remove extra spaces)
data.columns = data.columns.str.strip()

# Load label encoder for existing columns only
completed_routes = data.get('Completed More Than One Route', pd.Series())
comment_sentiments = data.get('Comment Sentiments', pd.Series())

if not completed_routes.empty and not comment_sentiments.empty:
    le.fit(completed_routes.unique().tolist() + comment_sentiments.unique().tolist())
else:
    print("Warning: 'Completed More Than One Route' or 'Comment Sentiments' column is missing in the processed data.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        input_data = request.get_json()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Map Ward/Branch values
        ward_mapping = {
            'Silver Berry Ward': 0, 'Crawford Plains Ward': 1, 'Lee Ridge Ward': 2, 'Griesbach Ward': 3,
            'Londonderry Ward': 4, 'Ellerslie Ward': 5, 'Blackmud Creek Ward': 6, 'Clareview Ward': 7,
            'Rutherford Ward': 8, 'Southgate Ward': 9, 'Forest Heights Ward': 10, 'Rabbit Hill Ward': 11,
            'Greenfield Ward': 12, 'Terwillegar Park Ward': 13, 'Namao Ward': 14, 'Woodbend Ward': 15,
            'Connors Hill Ward': 16, 'Stony Plain Ward': 17, 'Strathcona Married Student Ward': 18,
            'Rio Vista Ward': 19, 'Beaumont Ward': 20, 'Wild Rose Ward': 21, 'Drayton Valley Ward': 22,
            'Wainwright Branch': 23, 'Lago Lindo Branch': 24, 'Pioneer Ward': 25
        }
        input_df['Ward/Branch'] = input_df['Ward/Branch'].map(ward_mapping).fillna(-1)
        
        # Encode categorical features
        if 'Completed More Than One Route' in input_df.columns:
            input_df['Completed More Than One Route'] = le.transform([input_df['Completed More Than One Route']])[0]
        if 'Comment Sentiments' in input_df.columns:
            input_df['Comment Sentiments'] = le.transform([input_df['Comment Sentiments']])[0]

        # Make prediction
        prediction = model.predict(input_df)
        prediction_label = ['Negative', 'Positive', 'Neutral'][prediction[0]]

        # Return prediction result
        return jsonify({'prediction': prediction_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='127.0.0.1', port=port)
