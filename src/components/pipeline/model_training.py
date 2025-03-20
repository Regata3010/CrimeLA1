import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU


# def split_data(df):
#     train = df[df['DATE OCC'] < '2024-01-01']
#     test = df[df['DATE OCC'] >= '2024-01-01']
#     return train,test


class ModelTrainingConfig:
    model_path = os.path.join('artifacts','feature_processed_data.csv')
    
class ModelTraining:
    def __init__(self):
        self.model_training_config = ModelTrainingConfig()
        
    def initiate_model_training(self):
        try:
            logging.info("Model Training has started")
            df = pd.read_csv(self.model_training_config.model_path)
            
            #Split the data
            # train,test = split_data(df)
        
        except Exception as e:
            logging.info("Error in model training")
            raise CustomException(e,sys)
        
        
if __name__ == "__main__":
    obj = ModelTraining()
    obj.initiate_model_training()