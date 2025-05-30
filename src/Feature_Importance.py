import os
import joblib
import pandas as pd
import logging
import yaml
import mlflow
import mlflow.sklearn
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureAnalysis:
    def __init__(self, config_path="configs/predict_config.yaml"):
        """
        Initializes the Feature Analysis class using the configuration YAML.
        Uses the best model saved in `models/best_model.pkl` for analysis.
        """
        # Load configuration file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set paths from config
        self.data_path_1 = os.path.join("data", "processed", self.config["train_path"])  # Train data path
        self.data_path_2 = os.path.join("data", "processed", self.config["test_path"])  # Test data path
        self.model_path = os.path.join(self.config["models_dir"], "RandomForestClassifier.pkl")  # Best model path
        self.reports_dir = self.config["reports_dir"]
        self.feature_columns = self.config["feature_columns"]
        self.target_column = self.config["target_column"]
        self.mlflow_experiment = self.config["mlflow_experiment_name"]

        # Load Feature Importance flag (if enabled in config)
        self.feature_importance_enabled = self.config.get("feature_importance", False)

        # Initialize placeholders
        self.df = None
        self.model = None

        # Ensure reports directory exists
        os.makedirs(self.reports_dir, exist_ok=True)

        # Set MLflow experiment
        mlflow.set_experiment(self.mlflow_experiment)

    def load_data(self):
        """Step 1: Load dataset and trained best model."""
        try:
            self.df = pd.read_csv(self.data_path_1)
            logging.info(f"Step 1: Data loaded from {self.data_path_1} with {self.df.shape[0]} rows.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Step 1: Best model loaded from {self.model_path}.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def generate_feature_importance(self):
        """Step 2: Generate and save feature importance plot."""
        if not self.feature_importance_enabled:
            logging.info("Feature importance analysis is disabled in the configuration.")
            return

        if hasattr(self.model, "feature_importances_"):  # Tree-based models
            importance_values = self.model.feature_importances_
            logging.info("Feature importance extracted from `feature_importances_`.")
        elif hasattr(self.model, "coef_"):  # Linear models
            importance_values = self.model.coef_.flatten()
            logging.info("Feature importance extracted from `coef_` (linear model).")
        else:
            logging.warning("Model does not support feature importance analysis. Skipping this step.")
            return

        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({"Feature": self.feature_columns, "Importance": importance_values})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="Blues_r")
        plt.title("Feature Importance")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")

        # Ensure reports folder exists
        os.makedirs(self.reports_dir, exist_ok=True)

        # Save plot
        save_path = os.path.join(self.reports_dir, "feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Step 2: Feature importance plot saved at {save_path}.")

        # Track with MLflow
        with mlflow.start_run(run_name="Feature Importance Analysis"):
            mlflow.log_artifact(save_path)

        # Track with DVC
        #self.track_report_with_dvc(save_path)

        # Print feature importance
        print("\n=== Feature Importance ===")
        print(feature_importance_df)

    def correlation_heatmap(self):
        """Step 3: Generate and save correlation heatmap."""
        # Encode categorical columns to numeric if necessary
        if 'Ward/Branch' in self.df.columns:
            ward_mapping = {
                'Silver Berry Ward': 0, 'Crawford Plains Ward': 1, 'Lee Ridge Ward': 2, 'Griesbach Ward': 3,
                'Londonderry Ward': 4, 'Ellerslie Ward': 5, 'Blackmud Creek Ward': 6, 'Clareview Ward': 7,
                'Rutherford Ward': 8, 'Southgate Ward': 9, 'Forest Heights Ward': 10, 'Rabbit Hill Ward': 11,
                'Greenfield Ward': 12, 'Terwillegar Park Ward': 13, 'Namao Ward': 14, 'Woodbend Ward': 15,
                'Connors Hill Ward': 16, 'Stony Plain Ward': 17, 'Strathcona Married Student Ward': 18,
                'Rio Vista Ward': 19, 'Beaumont Ward': 20, 'Wild Rose Ward': 21, 'Drayton Valley Ward': 22,
                'Wainwright Branch': 23, 'Lago Lindo Branch': 24, 'Pioneer Ward': 25
            }
            self.df['Ward/Branch'] = self.df['Ward/Branch'].map(ward_mapping).fillna(-1)

        # Map "Yes" and "No" to numeric values for "Completed More Than One Route"
        if 'Completed More Than One Route' in self.df.columns:
            self.df['Completed More Than One Route'] = self.df['Completed More Than One Route'].replace({'Yes': 1, 'No': 0})

        # Map sentiment values to numeric
        if 'Comment Sentiments' in self.df.columns:
            sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            self.df['Comment Sentiments'] = self.df['Comment Sentiments'].map(sentiment_mapping).fillna(0)
        plt.figure(figsize=(10, 7))

        # Compute correlation matrix
        correlation_matrix = self.df[self.feature_columns + [self.target_column]].corr()

        # Plot heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Heatmap")

        # Ensure reports folder exists
        os.makedirs(self.reports_dir, exist_ok=True)

        # Save plot
        save_path = os.path.join(self.reports_dir, "correlation_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Step 3: Correlation heatmap saved at {save_path}.")

        # Track with MLflow
        with mlflow.start_run(run_name="Feature Correlation Analysis"):
            mlflow.log_artifact(save_path)

        # Track with DVC
        #self.track_report_with_dvc(save_path)


    def run_analysis(self):
        """Runs the full feature analysis pipeline."""
        logging.info("Starting feature analysis pipeline...")
        self.load_data()
        self.generate_feature_importance()
        self.correlation_heatmap()
        logging.info("Feature analysis completed.")


# Example Usage
if __name__ == "__main__":
    analyzer = FeatureAnalysis()
    analyzer.run_analysis()