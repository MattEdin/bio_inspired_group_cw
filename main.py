
import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("data/raw/diabetes_cleaned.csv")

    train_split_size = 0.7
    # last column is the binary classifier
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    
    X_train = X[:int(train_split_size * len(X))]
    y_train = y[:int(train_split_size * len(y))]
    X_test = X[int(train_split_size * len(X)):]
    y_test = y[int(train_split_size * len(y)):]

