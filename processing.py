import numpy as np
from sklearn.preprocessing import StandardScaler

feature_keys = [
    "Data.Temperature.Avg Temp", "Data.Temperature.Max Temp",
    "Data.Temperature.Min Temp", "Data.Wind.Direction", "Data.Wind.Speed"
]


def preprocess_data(df):
    # Select the features
    data = df[feature_keys].values

    # Standardize the features
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, scaler


def create_sequences(data, target_index, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length, target_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def split_data(X, y, split_ratio):
    split_index = int(len(X) * split_ratio)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    return X_train, X_val, y_train, y_val
