import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.data = None

    def load_data(self):
        """Load the dataset with a specified encoding."""
        if os.path.exists(self.raw_data_path):
            self.data = pd.read_csv(self.raw_data_path, encoding='latin1')
        else:
            self.data= None


    def calculate_time_spent(self):
        """Calculate the time spent in minutes."""
        self.data['The time you ended at'] = pd.to_datetime(self.data['The time you ended at'])
        self.data['The time you started at:'] = pd.to_datetime(self.data['The time you started at:'])
        self.data['Time Spent'] = (self.data['The time you ended at'] - self.data['The time you started at:']).dt.total_seconds() / 60
        self.data['Time Spent'] = self.data['Time Spent'].abs()
        
    def handle_missing_values(self):
        """Fill missing values in categorical and numerical columns."""
        self.data['Comment Sentiments'] = self.data['Comment Sentiments'].fillna('Neutral')
        self.data['Comments'] = self.data['Comments'].fillna('No Comments')
        
        num_cols = ['Number of routes completed', '# of Doors in Route', '# of Adult Volunteers in this route',
                    '# of Youth Volunteers in this route', '# of Donation Bags Collected/Route',
                    'Time to Complete (in minutes) pick up of bags /route', 'Time Spent']
        
        for col in num_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col].fillna(self.data[col].mean(), inplace=True)
        
    def clean_column_names(self):
        """Rename columns for clarity and drop unnecessary ones."""
        self.data.drop(columns=['The time you started at:', 'The time you ended at'], inplace=True)
        self.data.rename(columns={
            'Timestamp': 'Date',
            'ï»¿Drop Off Location': 'Drop Off Location',
            'City': 'City',
            'Stake': 'Stake',
            'Route Number/Name': 'Route',
            '# of Adult Volunteers in this route': '# of Adult Volunteers',
            '# of Youth Volunteers in this route': '# of Youth Volunteers',
            '# of Donation Bags Collected/Route': 'Donation Bags Collected',
            'Time to Complete (in minutes) pick up of bags /route': 'Time to Complete (min)',
            'Did you complete more than 1 route?': 'Completed More Than One Route',
            'Number of routes completed': 'Routes Completed',
            '# of Doors in Route': 'Doors in Route',
            'Comment Sentiments': 'Comment Sentiments',
            'Comments': 'Comments or Feedback'
        }, inplace=True)
    
    def convert_data_types(self):
        """Convert columns to appropriate data types."""
        int_columns = ['# of Adult Volunteers', '# of Youth Volunteers', 'Donation Bags Collected',
                       'Routes Completed', 'Doors in Route', 'Time to Complete (min)', 'Time Spent']
        self.data[int_columns] = self.data[int_columns].astype(int)
        
    def handle_outliers(self):
        """Cap and replace outliers in 'Time Spent'."""
        percentile_99 = self.data['Time Spent'].quantile(0.99)
        self.data['Time Spent'] = self.data['Time Spent'].clip(upper=percentile_99)
        threshold = 150  # minutes
        mean_time_spent = self.data[self.data['Time Spent'] <= threshold]['Time Spent'].mean()
        self.data.loc[self.data['Time Spent'] > threshold, 'Time Spent'] = mean_time_spent
    
    def save_data(self):
        """Save the cleaned data to the specified directory."""
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.data.to_csv(self.processed_data_path, index=False)
        print(f"Cleaned data saved at: {self.processed_data_path}")

    def preprocess(self):
        """Run all preprocessing steps."""
        self.load_data()
        if self.data is not None:
            self.calculate_time_spent()
            self.handle_missing_values()
            self.clean_column_names()
            self.convert_data_types()
            self.handle_outliers()
            self.save_data()
        else:
            print("No data to preprocess.")

# Example usage
if __name__ == "__main__":
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.join(PROJECT_ROOT_DIR, 'data/raw/Cleaned Proposed Data Collection.csv')
    processed_data_path = os.path.join(PROJECT_ROOT_DIR, 'data/processed/Cleaned_Proposed_Data_Collection.csv')
    processor = DataPreprocessor(raw_data_path, processed_data_path)
    processor.preprocess()
    print(PROJECT_ROOT_DIR)
    print(raw_data_path)
    print(processed_data_path)