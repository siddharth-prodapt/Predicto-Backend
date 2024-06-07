import pandas as pd
import json

def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None
