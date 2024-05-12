import pandas as pd
import keras
from zipfile import ZipFile

def load_data():
    # zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    # with ZipFile(zip_path, 'r') as zip_file:
    #     zip_file.extractall()
    csv_path = "weather.csv"
    df = pd.read_csv(csv_path)
    return df
