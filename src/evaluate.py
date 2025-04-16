import os
import joblib
import pandas as pd
import logging
import yaml
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Evaluator:
    def __init__(self, config_path="configs/predict_config.yaml"):
        """
        Initializes the Evaluator with paths to the test dataset and trained models.
        :param config_path: Path to the configuration YAML file.
        """
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            self.test_data_path = os.path.join("data", "processed", self.config["test_path"])
            self.models_dir = self.config["models_dir"]
            self.df = None
            self.models = {}
            self.metrics = {}

    def load_data(self):
        """Step 1: Load the processed test dataset."""
        try:
            logging.info(f"Loading test data from {self.test_data_path}...")
            self.df = pd.read_csv(self.test_data_path)
            logging.info(f"Test data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            raise

    def load_models(self):
        """Step 2: Load all trained models from the models directory."""
        try:
            if not os.path.exists(self.models_dir):
                logging.error(f"Models directory '{self.models_dir}' does not exist. Ensure models are trained first.")
                os.makedirs(self.models_dir, exist_ok=True)
                return

            model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pkl")]
            if not model_files:
                logging.error("No models found in the directory. Train models before evaluating.")
                return

            for model_file in model_files:
                model_path = os.path.join(self.models_dir, model_file)
                self.models[model_file] = joblib.load(model_path)

            logging.info(f"Step 2: Loaded {len(self.models)} models from {self.models_dir}. Available models: {model_files}")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise

    def evaluate_models(self):
        """Step 3: Evaluate each model on the test dataset."""
        if self.df is None:
            logging.error("Test data not loaded. Run load_data() first.")
            return
        if not self.models:
            logging.error("No models loaded. Run load_models() first.")
            return

        # Define features and target
        feature_columns = ['Ward/Branch', 'Completed More Than One Route', '# of Adult Volunteers', 'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        target_column = 'Comment Sentiments'

        if not all(col in self.df.columns for col in feature_columns + [target_column]):
            logging.error("Required feature columns are missing in the dataset.")
            return

        X_test = self.df[feature_columns]
        y_test = self.df[target_column]

        # Encode categorical columns
        categorical_columns = ['Ward/Branch', 'Completed More Than One Route']
        for col in categorical_columns:
            if col in X_test.columns:
                X_test[col] = pd.Categorical(X_test[col]).codes
        
        # Encode the target column
        sentiment_mapping = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
        y_test = y_test.map(sentiment_mapping)

        # Check for missing or invalid values in the target column
        if y_test.isnull().any():
            logging.error("Target column contains invalid or missing values after encoding.")
            return

        # Evaluate each model
        for model_name, model in self.models.items():
            logging.info(f"📊 Evaluating the model: {model_name}")
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                print(f"Accuracy for {model_name}: {accuracy:.4f}")
                print(f"Classification Report for {model_name}:\n{report}")

                self.metrics[model_name] = {"Accuracy Score": accuracy, "Classification Report": report}
            except Exception as e:
                logging.error(f"Error evaluating model {model_name}: {e}")
            
    logging.info("Step 3: Model evaluation complete.")

    def evaluate_pipeline(self):
        """Runs the full evaluation pipeline step by step."""
        logging.info("Starting evaluation pipeline...")
        self.load_data()
        self.load_models()
        self.evaluate_models()
        logging.info("Evaluation pipeline complete.")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate_pipeline()