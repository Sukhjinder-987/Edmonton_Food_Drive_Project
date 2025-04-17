import os
import joblib
import pandas as pd
import logging
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Predictor:
    def __init__(self, config_path: str):
        """
        Initializes the Predictor class with paths to the model and test data.
        :param config_path: Path to the configuration YAML file.
        """
        import yaml
        with open(config_path, "r") as path:
            self.config = yaml.safe_load(path)

        self.test_path = os.path.join("data", "processed", self.config["test_path"])
        self.model_path = os.path.join(self.config["models_dir"], "RandomForestClassifier.pkl")
        self.output_path = os.path.join("data","predicted_data", "predictions.csv")

        # Optionally, use MLflow autolog for sklearn
        mlflow.sklearn.autolog()

        # Set up mlflow experiment
        mlflow.set_experiment(self.mlflow_experiment_name)

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load_data(self):
        """Step 1: Loads the test data."""
        logging.info(f"Loading test data from {self.test_path}...")
        self.test_df = pd.read_csv(self.test_path)
        logging.info(f"Test data loaded with {self.test_df.shape[0]} rows and {self.test_df.shape[1]} columns.")

    def preprocess_data(self):
        """Step 2: Preprocesses the test data."""
        logging.info("Preprocessing test data...")
        feature_columns = ['Ward/Branch', 'Completed More Than One Route', '# of Adult Volunteers', 
                           'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        categorical_columns = ['Ward/Branch', 'Completed More Than One Route']
        numeric_columns = ['# of Adult Volunteers', 'Doors in Route', '# of Youth Volunteers', 'Time Spent']

        # Remove numeric columns with only missing values
        numeric_columns = [col for col in numeric_columns if self.test_df[col].notna().any()]
        missing_columns = set(['# of Adult Volunteers', 'Doors in Route', '# of Youth Volunteers', 'Time Spent']) - set(numeric_columns)
        if missing_columns:
            logging.warning(f"Skipping columns with only missing values: {missing_columns}")

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        self.test_df[numeric_columns] = imputer.fit_transform(self.test_df[numeric_columns])

        # Encode categorical columns
        for col in categorical_columns:
            self.test_df[col] = pd.Categorical(self.test_df[col]).codes

        # Scale numeric columns
        scaler = StandardScaler()
        self.test_df[numeric_columns] = scaler.fit_transform(self.test_df[numeric_columns])

        self.X_test = self.test_df[feature_columns]
        logging.info("Test data preprocessing complete.")

    def load_model(self):
        """Step 3: Loads the trained model."""
        logging.info(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        logging.info("Model loaded successfully.")

    def predict(self):
        """Step 4: Makes predictions on the test data."""
        logging.info("Making predictions...")
        predictions = self.model.predict(self.X_test)
        self.test_df["Predictions"] = predictions
        logging.info("Predictions complete.")

    def save_predictions(self):
        """Saves the predictions to a CSV file."""
        logging.info(f"Saving predictions to {self.output_path}...")
        self.test_df.to_csv(self.output_path, index=False)
        logging.info(f"Predictions saved to {self.output_path}.")

    def run(self):
        """Runs the full prediction pipeline."""
        logging.info("Starting training pipeline...")
        self.load_data()
        self.preprocess_data()
        self.load_model()
        self.predict()
        self.save_predictions()


if __name__ == "__main__":
    config_path = "configs/train_config.yaml"
    predictor = Predictor(config_path=config_path)
    predictor.run()