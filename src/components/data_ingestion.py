import os 
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import datetime
from datetime import datetime
from typing import Tuple



# do we actually need to split files in ingestion --> just check for better management
def convert_to_datetime(df,date_column:list):
    for date in date_column:
        df[date] = pd.to_datetime(df[date],errors='coerce',format='%m/%d/%Y %I:%M:%S %p')
    return df
    
def categorical_features(df):
    obj1 = df.select_dtypes(include=['object'])
    categorical_features = [item for item in obj1.columns]
    return categorical_features

def numerical_features(df):
    obj2 = df.select_dtypes(include=['int64','float64'])
    numerical_features = [item for item in obj2.columns]
    return numerical_features

def categorical_impute(df,categorical_features:list):
    cat_imputer = SimpleImputer(strategy='constant',fill_value='Unknown')
    df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
    return df

def numerical_impute(df,numerical_features:list):
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    return df

def check_missing_values(df):
    missing_columns = [col for col in df.columns if df[col].isna().sum() > 0]
    if missing_columns:
        logging.info(f"There are missing values in: {missing_columns}. Please check again.")
    else:
        logging.info("No missing values found.")
    return df  

def describe(df):
    logging.info(f"Shape of the DF is {df.shape}")
    logging.info(f"Missing Values : {df.isna().sum()}")
    logging.info(f"Dtypes of all Features: {df.dtypes}")
    logging.info(f"Before conversion: Unique values in DATE OCC: {df['DATE OCC'].dropna().unique()[:10]}")
    logging.info(f"Before conversion: Unique values in Date Rptd: {df['Date Rptd'].dropna().unique()[:10]}")
    return df

    
def filter_la_boundaries(df):
    """
    Filter DataFrame to include only records within Los Angeles city boundaries.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'LAT' and 'LON' columns
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only records within LA boundaries
        
    Raises:
        KeyError: If 'LAT' or 'LON' columns are missing
        ValueError: If DataFrame is empty
    """
    try:
        # Check if required columns exist
        required_columns = ['LAT', 'LON']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
            
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("Input DataFrame is empty")
            
        # Filter data within LA boundaries
        filtered_df = df[
            (df['LAT'] >= 33.7) & 
            (df['LAT'] <= 34.3) & 
            (df['LON'] >= -118.87) & 
            (df['LON'] <= -118.1)
        ]
        
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error in filter_la_boundaries: {str(e)}")
        raise CustomException(e, sys)
    
@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('src/artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")
        try:
            df = pd.read_csv(os.path.join('notebook/data', 'Crime_Data_from_2020_to_Present.csv'))
            logging.info("Data has been read successfully")
            columns_to_drop = ['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
            df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
            df = convert_to_datetime(df,['Date Rptd','DATE OCC'])
            categorical_cols = categorical_features(df)
            numerical_cols = numerical_features(df)
            df = categorical_impute(df, categorical_cols)
            df = numerical_impute(df, numerical_cols)
            df = filter_la_boundaries(df)
            df = check_missing_values(df)
            # os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data has been Converted into New and Cleaned Data")
            df = describe(df)
            logging.info("Reported all tasks complete")
     
           
        except Exception as e:
            logging.error("Error in loading data")
            raise CustomException(e, sys)
        
    
if __name__=='__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    
       
            