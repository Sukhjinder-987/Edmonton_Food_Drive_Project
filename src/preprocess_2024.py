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
            self.data = pd.read_csv(self.raw_data_path, encoding='utf-8-sig')

            if self.data.empty:
                raise ValueError(f"Data file at {self.raw_data_path} is empty.")
            print(f"Loaded data from: {self.raw_data_path}")
        else:
            raise FileNotFoundError(f"File not found at: {self.raw_data_path}")
    
    def clean_data(self):
        """Clean and transform the data."""
        if self.data is None or self.data.empty:
            raise ValueError("Data is empty. Cannot clean.")
        
        # Drop unnecessary columns
        drop_cols = ['ID','Email', 'Name', 'Additional Routes completed (2 routes)', 
                                            'Additional routes completed (3 routes)', 'Additional routes completed (3 routes)2',
                                            'Additional routes completed (More than 3 Routes)', 'Additional routes completed (More than 3 Routes)2', 
                                            'Additional routes completed (More than 3 Routes)3', 'Route Number/Name', 'Sherwood Park Stake']
        self.data.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Convert time columns and calculate 'Time Spent'  
    def calculate_time_spent(self):
        """Calculate the time spent in minutes."""
        self.data['Start time'] = pd.to_datetime(self.data['Start time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        self.data['Completion time'] = pd.to_datetime(self.data['Completion time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        self.data['Time Spent'] = (self.data['Completion time'] - self.data['Start time']).dt.total_seconds() / 60
        self.data.drop(columns=['Start time', 'Completion time'], inplace=True)

        
    def handle_missing_values(self):
    
        """Handle missing values in numeric and categorical columns."""
     
        # Handle categorical/text columns
        self.data['How many routes did you complete?'] = self.data['How many routes did you complete?'].fillna(1)
        self.data['Comment Sentiments'] = self.data['Comment Sentiments'].fillna('Neutral')
        self.data['Comments or Feedback'] = self.data['Comments or Feedback'].fillna('No Comments')
    
        # Handle numeric columns
        num_cols = [
            '# of Doors in Route', 
            '# of Adult Volunteers who participated in this route',
            '# of Youth Volunteers who participated in this route\n',
            '# of Donation Bags Collected'
        ]
        print(self.data.columns)
    
        for col in num_cols:

            if col in self.data.columns:

                try:
                    print(self.data[col])
                    self.data[col] = pd.to_numeric(self.data[col])
                    mean_value = self.data[col].mean()
                    self.data[col] = self.data[col].fillna(mean_value)

                except Exception as e:
                    print(f"Failed to process column '{col}': {e}")

            else:
                print(f"Warning: Column '{col}' not found in data.")

 
        #self.data['# of Doors in Route'] = pd.to_numeric(self.data['# of Doors in Route'], errors='coerce')
        
        #avg_doors_in_route = self.data['# of Doors in Route'].mean()
        #avg_time_spent = self.data['Time Spent'].mean()
        #avg_Adult_Volunteers_in_this_route = self.data['# of Adult Volunteers who participated in this route'].mean()
        #avg_Youth_Volunteers_in_this_route = self.data['# of Youth Volunteers who participated in this route\n'].mean()
        #avg_Donation_Bags_Collected = self.data['# of Donation Bags Collected'].mean()
        
        #self.data.fillna({
        #    '# of Doors in Route': avg_doors_in_route,
        #    'Time Spent': avg_time_spent,
        #    '# of Adult Volunteers who participated in this route': avg_Adult_Volunteers_in_this_route,
        #    '# of Youth Volunteers who participated in this route\n': avg_Youth_Volunteers_in_this_route,
        #    '# of Donation Bags Collected': avg_Donation_Bags_Collected
        #}, inplace=True)
        
    def clean_column_names(self):
        """Rename columns for clarity and drop unnecessary ones."""
        self.data.rename(columns={
            'Drop Off Location': 'Drop Off Location',
            'Stake': 'Stake',
            'Route Number/Name': 'Route',
            '# of Adult Volunteers who participated in this route': '# of Adult Volunteers',
            '# of Youth Volunteers who participated in this route\n': '# of Youth Volunteers',
            '# of Donation Bags Collected': 'Donation Bags Collected',
            'Did you complete more than 1 route?': 'Completed More Than One Route',
            'Comment Sentiments': 'Comment Sentiments',
            'Comments or Feedback': 'Comments or Feedback',
            'How many routes did you complete?': 'How many routes did you complete?',
            'Time Spent': 'Time Spent',
            '# of Doors in Route': 'Doors in Route',
            'Number of routes completed': 'Routes Completed',
            'COMBINED STAKES': 'Ward/Branch'    
        }, inplace=True)
        
    def convert_data_types(self):
        """Convert columns to appropriate data types."""
        int_columns = ['# of Adult Volunteers', '# of Youth Volunteers', 'Donation Bags Collected', 'Doors in Route']
        
        for col in int_columns:
            print(self.data.columns)
            self.data[col] = pd.to_numeric(self.data[col])
            mean_value = self.data[col].mean()
            self.data[col].fillna(round(mean_value), inplace=True)  # Fill with rounded mean value
            self.data[col] = self.data[col].astype(int)  # Convert back to integer

        #self.data['Time Spent for year 2024'] = pd.to_numeric(self.data['Time Spent for year 2024'], errors='coerce').fillna(0).astype(float)
        #self.data[int_columns] = self.data[int_columns].astype(int)
        #self.data['Time Spent for year 2024'] = self.data['Time Spent for year 2024'].astype(float)
        
        self.data = self.data.drop(columns=['Bonnie Doon Stake', 'Edmonton North Stake', 'Gateway Stake', 'Riverbend Stake', 'YSA Stake'])
        self.data['How many routes did you complete?'] = self.data['How many routes did you complete?'].replace('More than 3', '4')

    def save_cleaned_data(self):
        """Save the cleaned data to the specified directory."""
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        self.data.to_csv(self.processed_data_path, index=False)
        print(f"Cleaned data saved at: {self.processed_data_path}")

    def preprocess(self):
        """Run all preprocessing steps."""
        self.load_data()
        if self.data is not None:
            self.clean_data()
            self.calculate_time_spent()
            self.handle_missing_values()
            self.clean_column_names()
            self.convert_data_types()
            self.save_cleaned_data()
        else:
            print("No data to preprocess.")

if __name__ == "__main__":
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.join(PROJECT_ROOT_DIR, 'data/raw/Food_Drive_Data_Collection_2024.csv')
    processed_data_path = os.path.join(PROJECT_ROOT_DIR, 'data/processed/Cleaned_food_drive_data.csv')
    processor = DataPreprocessor(raw_data_path, processed_data_path)
    processor.preprocess()
    print(PROJECT_ROOT_DIR)
    print(raw_data_path)
    print(raw_data_path)

