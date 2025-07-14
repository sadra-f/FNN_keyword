import pandas as pd

def read_csv_dataset(path):
    dataset = pd.read_csv(path)
    return dataset
