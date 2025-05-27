# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os 
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)
    
    # Step 1: Load the model from the specified path
    model = mlflow.sklearn.load_model(args.model_path)

    # Step 2: Log the model
    artifact_path = "random_forest_price_regressor"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )

    # Step 3: Register the model using its artifact URI
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow_model = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    model_version = mlflow_model.version

    # Step 4: Save model registration details
    print("Writing JSON")
    model_info = {"id": f"{args.model_name}:{model_version}"}
    output_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(model_info, of)


if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()