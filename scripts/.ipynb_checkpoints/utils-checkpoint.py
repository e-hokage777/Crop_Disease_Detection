from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def get_labelencoder(filepath, column):
    df = pd.read_csv(filepath)
    label_encoder = LabelEncoder()
    values = df[column].values.tolist()
    values.append("background")
    label_encoder.fit(values)
    return label_encoder

def get_file_dir():
    '''
    function to get the working directory of the currently calling file
    '''
    return os.path.dirname(os.path.abspath(__file__))