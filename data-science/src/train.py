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
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument('--n_estimators', type=int, default=100,
                    help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')
    parser.add_argument("--model_output", type=str, help="Path of output model")

    args = parser.parse_args()

    return args

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods. 
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data)) 
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.
    y_train = train_df["price"].values
    X_train = train_df.drop("price", axis=1).values
    y_test = test_df["price"].values
    X_test = test_df.drop("price", axis=1).values  
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.
    tree_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
    tree_model = tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)  
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)  
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.
    mse = mean_squared_error(y_test, tree_predictions)
    print('MSE of Random Forest Regressor on test set: {:.2f}'.format(mse))  
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  
    mlflow.log_metric("MSE", float(mse))  
    # Output the model
    mlflow.sklearn.save_model(tree_model, args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

