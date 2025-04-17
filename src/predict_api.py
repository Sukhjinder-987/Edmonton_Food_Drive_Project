import os
import joblib
import pandas as pd
import logging
import yaml
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

# Configure logging
# Configure logging
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "predict_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_app.api") 

app = Flask(__name__)

# Load configuration
config_path = "configs/predict_config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_v1_path = config.get("model_v1_path")
model_v2_path = config.get("model_v2_path")

if not model_v1_path or not os.path.exists(model_v1_path):
    raise FileNotFoundError(f"Model v1 file not found: {model_v1_path}")

if not model_v2_path or not os.path.exists(model_v2_path):
    raise FileNotFoundError(f"Model v2 file not found: {model_v2_path}")

# Load models
model_v1 = joblib.load(model_v1_path)
model_v2 = joblib.load(model_v2_path)

@app.route('/food_drive_home', methods=['GET'])
def home():
    """Home endpoint providing API usage information"""
    info = {
        "description": "This API serves machine learning models for predicting donation bags collected.",
        "endpoints": {
            "/v1/predict": "Predict using model version 1",
            "/v2/predict": "Predict using model version 2",
            "/health_status": "Check the health status of the API"
        },
        "request_format": {
            "Ward/Branch": "string",
            "Completed More Than One Route": "boolean",
            "# of Adult Volunteers": "int",
            "# of Youth Volunteers": "int",
            "Doors in Route": "int",
            "Time_spent": "float"
        }
    }
    return jsonify(info)

@app.route('/health_status', methods=['GET'])
def health_status():
    """Health endpoint to confirm API is running"""
    return jsonify({"status": "API is running and operational"})

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict endpoint using model version 1"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])

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
        df['Ward/Branch'] = df['Ward/Branch'].map(ward_mapping).fillna(-1)

        # Map "Yes" and "No" to numeric values for "Completed More Than One Route"
        if 'Completed More Than One Route' in df.columns:
            df['Completed More Than One Route'] = df['Completed More Than One Route'].replace({'Yes': 1, 'No': 0})

        # Ensure all required columns are present
        required_columns = ['Ward/Branch', 'Completed More Than One Route', '# of Adult Volunteers', 
                            'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing columns

        # Reorder columns to match the training data
        df = df[required_columns]

        # Make prediction
        prediction = model_v1.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])

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
        df['Ward/Branch'] = df['Ward/Branch'].map(ward_mapping).fillna(-1)

        # Map "Yes" and "No" to numeric values for "Completed More Than One Route"
        if 'Completed More Than One Route' in df.columns:
            df['Completed More Than One Route'] = df['Completed More Than One Route'].replace({'Yes': 1, 'No': 0})

        # Ensure all required columns are present
        required_columns = ['Ward/Branch', 'Completed More Than One Route', '# of Adult Volunteers', 
                            'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing columns

        # Reorder columns to match the training data
        df = df[required_columns]

        # Make prediction
        prediction = model_v2.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)