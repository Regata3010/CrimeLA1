from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from src.components.feature_engineering import create_basic_engineering

def crimes_per_year(df):
    crimes_per_year = df['Year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=crimes_per_year.index, y=crimes_per_year.values, palette='viridis')
    plt.title('Number of Crimes per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.show()
    
def crimes_per_area(df):
    area_columns = [col for col in df.columns if 'AREA NAME_' in col]
    crimes_per_area = df[area_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=crimes_per_area.index, y=crimes_per_area.values, palette='viridis')
    plt.title('Crimes per Area')
    plt.xlabel('Area')
    plt.ylabel('Number of Crimes')
    plt.show()

def victim_sex_barplot(df):
    victim_columns = [col for col in df.columns if 'Victim Sex' in col]
    victims_count = df[victim_columns].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=victims_count.index, y=victims_count.values, palette='viridis')
    plt.title('Victim Sex')
    plt.xlabel('Sex')
    plt.ylabel('Number of Victims')
    plt.show()
    
def crime_size_bymonths(df):
    labels = ['Jan','Feb','March','April','May','June','July','August','September','October','November','December']
    monthly_crimes = df['Month'].value_counts().sort_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=monthly_crimes.index, y=monthly_crimes.values, palette='viridis')
    plt.title('Crime Size by Months')
    plt.xlabel('Months')
    plt.ylabel('Number of Crimes')
    plt.xticks(monthly_crimes.index, labels)
    plt.show()
    
def plot_kdeplot(df):
    plt.figure(figsize=(20,10))
    sns.kdeplot(df['LAT'],df['LON'],shade=True,color='red',bw_adjust=0.5)
    plt.title('Crime Density in Los Angeles')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()
    
def line_plot_crime_trends(df):
    plt.figure(figsize=(10,6))
    plt.plot(df['DATE OCC'],df['Crm Cd'],color='red')
    plt.title('Crime Trends in Los Angeles')
    plt.xlabel('Date')
    plt.ylabel('Number of Crimes')
    plt.show()

def crimes_by_timeofday(df):
    daily_night_crimes = df.groupby(df['DATE OCC'].dt.date)['Is_Night'].sum()
    daily_day_crimes = df.groupby(df['DATE OCC'].dt.date)['Is_peak_hour'].sum()
    
    plt.figure(figsize=(10,6))
    plt.stackplot(daily_night_crimes.index, 
                 [daily_night_crimes.values, daily_day_crimes.values],
                 labels=['Night','Day'])
    plt.title('Crimes by Time of Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Crimes')
    plt.legend()
    plt.show()
    
    
    
    
    
    
class DataVisualizationConfig:
    data_path: str = os.path.join('artifacts', "data.csv")
    

class DataVisualization:
    def __init__(self):
        self.visualization_config = DataVisualizationConfig()
        
    def initiate_data_visualization(self):
        logging.info("Data Visualization has started")
        try:
            df = pd.read_csv(self.visualization_config.data_path)
            logging.info("Data has been read successfully")
            
            #Now begin with Visualization
            logging.info("Visualization has started")
            crimes_per_year(df)
            crimes_per_area(df)
            victim_sex_barplot(df)
            crime_size_bymonths(df)
            plot_kdeplot(df)
            line_plot_crime_trends(df)
            crimes_by_timeofday(df)
            logging.info("Visualization has completed")
            
        except Exception as e:
            logging.error("Error in data visualization")
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = DataVisualization()
    obj.initiate_data_visualization()