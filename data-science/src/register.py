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
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model and registers it to MLflow'''

    model_name = "used-cars-model"
    print("Registering model:", model_name)

    # Step 1: Load and log the model to MLflow
    model = mlflow.sklearn.load_model(args.model_path)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    # Step 2: Construct the URI to the logged model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

    # Step 3: Register the model under the specified name
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"✅ Registered model: {registered_model.name}, version: {registered_model.version}")

    # Step 4: Save model info to output JSON
    model_info = {
        "model_name": registered_model.name,
        "model_version": registered_model.version
    }

    output_dir = Path(args.model_info_output_path).parent
    os.makedirs(output_dir, exist_ok=True)

    with open(args.model_info_output_path, "w") as f:
        json.dump(model_info, f)

    print(f"📦 Model info saved to: {args.model_info_output_path}")

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")

    main(args)
    mlflow.end_run()
