from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import datetime
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype

def convert_to_datetime(df, date_column: list):
    df = df.copy()
    for date in date_column:
        if not is_datetime64_any_dtype(df[date]):
            try:
                # First try the original format
                df[date] = pd.to_datetime(df[date], format='%m/%d/%Y %I:%M:%S %p')
            except ValueError:
                # If that fails, try parsing without format (will handle ISO format)
                df[date] = pd.to_datetime(df[date])
            logging.info(f"Converted {date} to datetime. Sample values: {df[date].head()}")
    return df
    
def create_basic_engineering(df):
    df = df.copy()
    df['Year'] = df['DATE OCC'].dt.year
    df['Month'] = df['DATE OCC'].dt.month
    df['Day'] = df['DATE OCC'].dt.day
    df['Quarter'] = df['DATE OCC'].dt.quarter
    df['Hour'] = df['DATE OCC'].dt.hour
    df['Day_of_week'] = df['DATE OCC'].dt.dayofweek
    df['Is_weekend'] = df['Day_of_week'].isin([5, 6]).astype(int)
    df['Is_Night'] = ((df['Hour'] >= 20) | (df['Hour'] <= 5)).astype(int)
    df['Is_peak_hour'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    
    logging.info(f"Basic features created. Shape: {df.shape}")
    logging.info(f"Sample datetime after basic engineering: {df['DATE OCC'].head()}")
    return df

    

def monthly_create_lag_features(df, target_col, lag_features:list):
    df = df.copy()
    
    # Verify datetime
    if not is_datetime64_any_dtype(df['DATE OCC']):
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
    
    # Create Year-Month string
    df['Year_Month'] = df['DATE OCC'].dt.strftime('%Y-%m') #better parsing of date values to group them
    unique_months = df['Year_Month'].nunique()
    logging.info(f"Number of unique months: {unique_months}")
    logging.info(f"Sample Year_Month values: {df['Year_Month'].head()}")
    
    # Calculate monthly crime counts
    monthly_counts = df.groupby('Year_Month').size().reset_index(name='Crime_Count')
    df = df.merge(monthly_counts, on='Year_Month', how='left')
    
    # Create lag features by month
    df = df.sort_values(['Year_Month', 'DATE OCC'])
    for lag in lag_features:
        df[f'{target_col}_lag_{lag}'] = df.groupby('Year_Month')[target_col].shift(lag)
    
    # Fill missing values
    lag_columns = [f'{target_col}_lag_{lag}' for lag in lag_features]
    df[lag_columns] = df[lag_columns].fillna(0)
    
    logging.info(f"Lag features created. Shape: {df.shape}")
    logging.info(f"Sample datetime after lag features: {df['DATE OCC'].head()}")
    return df

def create_rolling_features(df, target_col1, windows:list):
    df = df.copy()
    df = df.sort_values('DATE OCC')
    
    for window in windows:
        col_name = f'{target_col1}_rolling_mean_{window}'
        df[col_name] = df[target_col1].rolling(window=window, min_periods=1).mean()
        df[col_name] = df[col_name].fillna(df[col_name].mean())
        
    logging.info(f"Rolling features created. Shape: {df.shape}")
    logging.info(f"Sample datetime after rolling features: {df['DATE OCC'].head()}")
    return df

def create_FE_pipeline(df):
    try:
        logging.info("Starting feature engineering pipeline")
        logging.info(f"Initial shape: {df.shape}")
        logging.info(f"Initial date range: {df['DATE OCC'].min()} to {df['DATE OCC'].max()}")
        
        # Convert dates
        df = convert_to_datetime(df, ['Date Rptd', 'DATE OCC'])
        
        # Basic features
        df = create_basic_engineering(df)
        
        # Lag features
        df = monthly_create_lag_features(df, 'Crm Cd', lag_features=[1, 3, 6])
        
        # Rolling features
        df = create_rolling_features(df, 'Crime_Count', windows=[6, 12])
        
        # Final checks
        logging.info(f"Final shape: {df.shape}")
        logging.info(f"Final date range: {df['DATE OCC'].min()} to {df['DATE OCC'].max()}")
        logging.info(f"Final columns: {df.columns.tolist()}")
        
        # Verify no NaN in critical columns
        critical_cols = ['DATE OCC', 'Date Rptd', 'Crime_Count']
        for col in critical_cols:
            na_count = df[col].isna().sum()
            logging.info(f"NaN count in {col}: {na_count}")
            
        return df
        
    except Exception as e:
        logging.error(f"Error in feature engineering pipeline: {str(e)}")
        raise CustomException(e, sys)

def split_train_test(df, split_date:datetime):
    try:
        df = df.copy()
        logging.info(f"Split function - Initial shape: {df.shape}")
        logging.info(f"Split function - Date range: {df['DATE OCC'].min()} to {df['DATE OCC'].max()}")
        
        # Parse split date
        if isinstance(split_date, str):
            split_date = pd.to_datetime(split_date)
        logging.info(f"Split date: {split_date}")
        
        # Ensure DATE OCC is datetime
        if not is_datetime64_any_dtype(df['DATE OCC']):
            df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
        
        # Create splits
        train_set = df[df['DATE OCC'] < split_date].copy()
        test_set = df[df['DATE OCC'] >= split_date].copy()
        
        logging.info(f"Training set: {len(train_set)} records, {train_set['DATE OCC'].min()} to {train_set['DATE OCC'].max()}")
        logging.info(f"Test set: {len(test_set)} records, {test_set['DATE OCC'].min()} to {test_set['DATE OCC'].max()}")
        
        return train_set, test_set
        
    except Exception as e:
        logging.error(f"Error in train-test split: {str(e)}")
        raise CustomException(e, sys)

@dataclass
class FeatureEngineeringConfig:
    initial_data_path = os.path.join('src/artifacts','data.csv')
    train_data_path = os.path.join('src/artifacts','train.csv')
    test_data_path = os.path.join('src/artifacts','test.csv')  
    
class FeatureEngineering:
    def __init__(self):
        self.feature_engineering_config = FeatureEngineeringConfig()
        
    def initiate_feature_engineering(self):
        try:
            logging.info("Feature Engineering started")
            
            # Read data
            df = pd.read_csv(self.feature_engineering_config.initial_data_path)
            
            # Initial datetime conversion - use flexible parsing
            try:
                df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
                df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
            except Exception as e:
                logging.error(f"Error in datetime conversion: {str(e)}")
                raise CustomException(e, sys)
            
            logging.info(f"Initial shape: {df.shape}")
            logging.info(f"Initial date range: {df['DATE OCC'].min()} to {df['DATE OCC'].max()}")
            logging.info(f"Sample dates: \nDATE OCC: {df['DATE OCC'].head()}\nDate Rptd: {df['Date Rptd'].head()}")
            
            # Apply feature engineering
            df = create_FE_pipeline(df)
            
            # Ensure dates are still datetime before saving
            if not is_datetime64_any_dtype(df['DATE OCC']):
                df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
            if not is_datetime64_any_dtype(df['Date Rptd']):
                df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
            
            # Save processed data before splitting
            processed_path = os.path.join('src/artifacts', 'feature_processed_data.csv')
            df.to_csv(processed_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
            
            # Split and save train/test sets
            train_set, test_set = split_train_test(df, '2024-01-01')
            
            # Save splits with date columns preserved
            train_set.to_csv(self.feature_engineering_config.train_data_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
            test_set.to_csv(self.feature_engineering_config.test_data_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
            
            logging.info("Feature Engineering completed successfully")
            return processed_path
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = FeatureEngineering()
    obj.initiate_feature_engineering()
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# def handle_invalid_dates(df, date_columns, method='drop', fill_value='2020-01-01'):
    # """
    # Handles missing or invalid values in date columns.

    # Args:
    #     df (pd.DataFrame): The input DataFrame.
    #     date_columns (list): List of date columns to check.
    #     method (str): 'drop' to remove rows with NaT, 'fill' to replace with a specific date.
    #     fill_value (str): The fallback date to use when filling NaT values.

    # Returns:
    #     pd.DataFrame: The cleaned DataFrame.
    # """
    # for date_column in date_columns:
    #     missing_before = df[date_column].isna().sum()
    #     logging.info(f"Missing values in {date_column} before handling: {missing_before}")

    #     if method == 'drop':
    #         df = df.dropna(subset=[date_column])  
    #     elif method == 'fill':
    #         df[date_column] = df[date_column].fillna(pd.to_datetime(fill_value))  

    #     missing_after = df[date_column].isna().sum()
    #     logging.info(f"Missing values in {date_column} after handling: {missing_after}")

    # return df