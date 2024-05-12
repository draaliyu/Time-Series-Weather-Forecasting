import pandas as pd
from load_data import load_data
from processing import *
from training import build_model, train_model, plot_history, save_model

# Parameters
#URI = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
SEQ_LENGTH = 144
TARGET_INDEX = 0  # Index for "T (degC)"
SPLIT_RATIO = 0.8
EPOCHS = 20
BATCH_SIZE = 2
MODEL_PATH = 'trained_model.h5'

def main():
    # Step 1: Load the data
    df = load_data()

    # Step 2: Preprocess the data
    data, scaler = preprocess_data(df)

    # Step 3: Create sequences
    X, y = create_sequences(data, TARGET_INDEX, SEQ_LENGTH)

    # Step 4: Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y, SPLIT_RATIO)

    # Step 5: Build and train the model
    model = build_model(SEQ_LENGTH, len(feature_keys))
    history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

    # Step 6: Plot training history
    plot_history(history)

    # Step 7: Save the trained model
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
