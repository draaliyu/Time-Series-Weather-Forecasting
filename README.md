# Time Series Forecasting Model

This repository contains scripts to load, preprocess, and train a time series forecasting model using climate data. The dataset was accessed from the [CORGIS Dataset Project](https://corgis-edu.github.io/corgis/csv/weather/).

## Features Used

The following features from the dataset are used for training the model:

- Data.Temperature.Avg Temp
- Data.Temperature.Max Temp
- Data.Temperature.Min Temp
- Data.Wind.Direction
- Data.Wind.Speed

The target variable is:

- Data.Temperature.Avg Temp

## Scripts Overview

1. **load_data.py**: Handles downloading and extracting the dataset.
2. **processing.py**: Handles data preprocessing and sequence creation.
3. **training.py**: Builds and trains the model.
4. **main.py**: Coordinates the entire workflow.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- sklearn
- tensorflow
- keras

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow keras
```
## Running the Scripts

1. **Download and Load Data**: The dataset will be downloaded and extracted automatically by the load_data.py script.

2. **Preprocess Data**: The processing.py script will preprocess the data, normalize the features, and create time series sequences.

3. **Train the Model**: The training.py script will build and train an LSTM-based model using the preprocessed data.

4. **Main Script**: The main.py script coordinates all the steps. Run this script to execute the entire workflow.

## Example Usage

To run the entire process, simply execute the main.py script:
```
python main.py
```
