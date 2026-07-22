import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SignalEDA:
    """
    Initial Exploratory Data Analysis (EDA) module for time-series signals.
    Handles data profiling, missing values, data types, and global outlier detection.
    """
    def __init__(self, filepath: str):
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully. Shape: {self.df.shape}")

    def describe_data(self):
        """Prints standard statistical descriptions and dataframe information."""
        print("\n### Data Description")
        print(self.df.describe())
        
        print("\n### Data Information")
        self.df.info()

    def visualize_signal(self, column_name: str = None):
        """
        Plots a given time-series signal. 
        Defaults to the first column if no name is provided.
        """
        print("\n### Data Visualization")
        
        # Default to the first column if none is specified
        if column_name is None:
            column_name = self.df.columns[0]
            
        if column_name not in self.df.columns:
            print(f"Error: Column '{column_name}' not found in dataset.")
            return

        # Configure the plot size
        plt.figure(figsize=(6, 3))

        # Plot the signal
        plt.plot(self.df[column_name], color='blue', linewidth=0.4, label=column_name)

        # Add formatting and labels
        plt.title(f'{column_name} Time-Series Plot', fontsize=10, fontweight='bold')
        plt.xlabel('Index', fontsize=8)
        plt.ylabel('Signal Value', fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.25)

        # Render the line graph
        plt.tight_layout()
        plt.show()

    def check_null_values(self):
        """Checks for and counts any missing (null) values in the dataset."""
        print("\n### Null Values Count")
        null_counts = self.df.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if not null_cols.empty:
            null_df = pd.DataFrame({'Column': null_cols.index, 'Null Count': null_cols.values})
            print("The dataset has the following null values (as a table):")
            print(null_df.to_string(index=False))
        else:
            print("The dataset has no null values.")

    def check_type_formats(self):
        """Analyzes the data types of the columns to ensure consistency."""
        print("\n### Type Formats")
        print("Data types of each column:")
        types_df = pd.DataFrame(self.df.dtypes, columns=['Data Type'])
        print(types_df.head(10).to_string()) # Print head to avoid console spam for 500 columns
        print("...")

        # Check if all types are the same
        unique_types = self.df.dtypes.unique()
        if len(unique_types) == 1:
            print(f"\nAll columns have the same data type: {unique_types[0]}")
        else:
            print("\nColumns have different data types.")

    def detect_global_outliers(self):
        """
        Detects global outliers across all numerical columns using the IQR method.
        Calculates average lower and upper thresholds for the entire dataset.
        """
        print("\n### Outlier Values (IQR method)")
        
        numerical_cols = self.df.select_dtypes(include=['number']).columns

        if not numerical_cols.empty:
            all_lower_bounds = []
            all_upper_bounds = []

            for col in numerical_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                all_lower_bounds.append(lower_bound)
                all_upper_bounds.append(upper_bound)

            # Calculate the global average outlier bounds
            global_lower_threshold = sum(all_lower_bounds) / len(all_lower_bounds)
            global_upper_threshold = sum(all_upper_bounds) / len(all_upper_bounds)

            # Count values less than the global lower threshold and more than the global upper threshold
            total_less_than_global_lower = 0
            total_more_than_global_upper = 0

            for col in numerical_cols:
                total_less_than_global_lower += (self.df[col] < global_lower_threshold).sum()
                total_more_than_global_upper += (self.df[col] > global_upper_threshold).sum()

            print(f"The dataset has {total_less_than_global_lower} values less than the general lower outlier threshold ({global_lower_threshold:.2f}).")
            print(f"The dataset has {total_more_than_global_upper} values more than the general upper outlier threshold ({global_upper_threshold:.2f}).")
        else:
            print("No numerical columns found to check for outliers.")

    def run_full_eda(self):
        """Pipeline method to execute all EDA steps sequentially."""
        self.describe_data()
        self.check_null_values()
        self.check_type_formats()
        self.detect_global_outliers()
        
        # Visualize the first column by default
        first_col = self.df.columns[0]
        self.visualize_signal(column_name=first_col)


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create a dummy CSV if it doesn't exist to ensure the script runs out of the box
    filename = 'data/signals_data.csv'

    # Initialize the EDA class
    eda = SignalEDA(filename)
    
    # Run the full pipeline
    eda.run_full_eda()