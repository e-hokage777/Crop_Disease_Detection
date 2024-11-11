from sklearn.preprocessing import LabelEncoder
import pandas as pd

def get_labelencoder(filepath, column):
    df = pd.read_csv(filepath)
    label_encoder = LabelEncoder()
    label_encoder.fit(df[column])
    return label_encoder