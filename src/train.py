import os
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
import joblib
import subprocess
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class with paths to processed data and model saving.

        :param config_path: Path to the configuration YAML file.
        """
        self.config_path = config_path

        # Load config file
        with open(config_path, "r") as path:
            self.config = yaml.safe_load(path)

        # Optionally, use MLflow autolog for sklearn
        mlflow.sklearn.autolog()

        # Start an MLflow run using the context manager
        with mlflow.start_run() as run:
            # Log training parameters
            mlflow.log_params(self.config)

        self.train_path = os.path.join("data","processed", self.config["train_path"])
        self.test_path = os.path.join("data","processed", self.config["test_path"])   

        self.models_dir = self.config["models_dir"]
        self.train_df = None
        self.test_df = None
        self.models = {
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=self.config["random_forest"]["n_estimators"],
                max_depth=self.config["random_forest"]["max_depth"],
            ),
            #"KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier()       
        }
        self.param_grids = self.config["param_grids"]
        self.results_regular = []  # Store results for default models
        self.results_tuned = []  # Store results for tuned models

        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Start MLflow Experiment
        mlflow.set_experiment(self.config["mlflow_experiment_name"])

        # Enable MLflow autologging for Scikit-learn models
        mlflow.sklearn.autolog()

    def load_data(self):
        """Step 1: Load the training and testing datasets."""
        try:
            self.train_df = pd.read_csv(self.train_path)
            self.test_df = pd.read_csv(self.test_path)
            logging.info(f"Step 1: Training data loaded from {self.train_path} ({self.train_df.shape[0]} rows).")
            logging.info(f"Step 1: Testing data loaded from {self.test_path} ({self.test_df.shape[0]} rows).")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise


    def prepare_data(self):
        """Step 3: Prepare features and target for model training."""
        # Preprocess the feature matrix (2023 data)
        feature_columns = ['Ward/Branch', 'Completed More Than One Route', '# of Adult Volunteers', 'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        target_column = 'Comment Sentiments'

        # Check for missing columns in train and test datasets
        missing_train_columns = [col for col in feature_columns if col not in self.train_df.columns]
        missing_test_columns = [col for col in feature_columns if col not in self.test_df.columns]

        if missing_train_columns:
            raise KeyError(f"Missing columns in training data: {missing_train_columns}")
        if missing_test_columns:
            raise KeyError(f"Missing columns in testing data: {missing_test_columns}")

        self.X_train = self.train_df[feature_columns]
        self.y_train = self.train_df[target_column]
        self.X_test = self.test_df[feature_columns]
        self.y_test = self.test_df[target_column]

        # Separate numeric and categorical columns
        numeric_columns = ['# of Adult Volunteers', 'Doors in Route', '# of Youth Volunteers', 'Time Spent']
        categorical_columns = ['Ward/Branch', 'Completed More Than One Route']

    
        # Handle missing values
        #imputer = SimpleImputer(strategy="mean")
        #self.X_train = pd.DataFrame(imputer.fit_transform(self.X_train), columns=feature_columns)
        #self.X_test = pd.DataFrame(imputer.transform(self.X_test), columns=feature_columns)

        # Encode categorical columns using pandas.Categorical
        for col in categorical_columns:
            combined_categories = pd.concat([self.X_train[col], self.X_test[col]]).unique()
            self.X_train.loc[:, col] = pd.Categorical(self.X_train[col], categories=combined_categories).codes
            self.X_test.loc[:, col] = pd.Categorical(self.X_test[col], categories=combined_categories).codes

        # Encode categorical columns using pandas.Categorical
        #for col in ['Ward/Branch', 'Completed More Than One Route']:
            #combined_categories = pd.concat([self.X_train[col], self.X_test[col]]).unique()
            #self.X_train.loc[:, col] = pd.Categorical(self.X_train[col], categories=combined_categories).codes
            #self.X_test.loc[:, col] = pd.Categorical(self.X_test[col], categories=combined_categories).codes

        # Scale the numeric columns
        scaler = StandardScaler()
        self.X_train[numeric_columns] = scaler.fit_transform(self.X_train[numeric_columns])
        self.X_test[numeric_columns] = scaler.transform(self.X_test[numeric_columns])

        # Scale the feature columns
        #scaler = StandardScaler()
        #self.X_train[feature_columns] = scaler.fit_transform(self.X_train[feature_columns])
        #self.X_test[feature_columns] = scaler.transform(self.X_test[feature_columns])
    

        # Encode the target column
        sentiment_mapping = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
        self.y_train = self.y_train.map(sentiment_mapping)
        self.y_test = self.y_test.map(sentiment_mapping)

        logging.info("Step 3: Data prepared for training.")

    def train_model(self, model_name, model, tuned=False):
        """Train a model and log results."""
        with mlflow.start_run(run_name=f"{model_name}_Tuned" if tuned else model_name):
            model.fit(self.X_train, self.y_train)

            # Make predictions
            y_pred = model.predict(self.X_test)

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)

            # Store results
            result = {
                "Model": model_name + ("_Tuned" if tuned else ""),
                "Accuracy": accuracy,
            }

            if tuned:
                self.results_tuned.append(result)
            else:
                self.results_regular.append(result)

            # Log results
            logging.info(f"{model_name} - Accuracy: {accuracy:.4f}")

            # Save the model
            model_save_path = os.path.join(self.models_dir, f"{result['Model']}.pkl")
            joblib.dump(model, model_save_path)
            logging.info(f"Model '{result['Model']}' saved to {model_save_path}.")

    def hypertune_model(self, model_name, model):
        """Perform hyperparameter tuning using GridSearchCV."""
        if model_name not in self.param_grids:
            return model

        logging.info(f"Hyperparameter tuning for {model_name}...")
        param_grid = self.param_grids[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        logging.info(f"Best hyperparameters for {model_name}: {best_params}")

        return grid_search.best_estimator_

    def save_model(self):
        """Train models with and without hyperparameter tuning."""
        os.makedirs(self.models_dir, exist_ok=True)

        for model_name, model in self.models.items():
            logging.info(f"Training {model_name} without tuning...")
            self.train_model(model_name, model, tuned=False)

            if model_name in self.param_grids:
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)

    def train_pipeline(self):
        """Run the full training pipeline."""
        logging.info("Starting training pipeline...")
        self.load_data()
        self.prepare_data()

        # Train each model
        for model_name, model in self.models.items():
            logging.info(f"Training {model_name} without tuning...")
            self.train_model(model_name, model, tuned=False)

            # Perform hyperparameter tuning if applicable
            if model_name in self.param_grids:
                logging.info(f"Hyperparameter tuning for {model_name}...")
                tuned_model = self.hypertune_model(model_name, model)
                self.train_model(model_name, tuned_model, tuned=True)
                
        self.train_model()
        self.hypertune_model()
        self.save_model()
        logging.info("Training pipeline complete.")

if __name__ == "__main__":
    trainer = Trainer(config_path="configs/train_config.yaml")
    trainer.train_pipeline()
