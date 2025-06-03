# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)

    return parser.parse_args()

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Load datasets
    train_df = pd.read_csv(Path(args.train_data) / "train.csv")
    test_df = pd.read_csv(Path(args.test_data) / "test.csv")


    # Split into features and labels
    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Log parameters and metric
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("MSE", mse)

    # Save model
    output_path = Path(args.model_output)
    output_path.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, path=str(output_path))

if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print(f"Train dataset input path: {args.train_data}")
    print(f"Test dataset input path: {args.test_data}")
    print(f"Model output path: {args.model_output}")
    print(f"Number of Estimators: {args.n_estimators}")
    print(f"Max Depth: {args.max_depth}")

    main(args)

    mlflow.end_run()
