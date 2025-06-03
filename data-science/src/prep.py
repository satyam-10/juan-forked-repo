import argparse
import os
from pathlib import Path
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--raw_data", type=str, help="Path to raw data (mounted folder or CSV file)")
    parser.add_argument("--train_data", type=str, help="Output path for training data")
    parser.add_argument("--test_data", type=str, help="Output path for test data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test/train ratio")
    return parser.parse_args()

def main(args):
    # Locate the CSV file from directory or direct path
    if os.path.isdir(args.raw_data):
        csv_files = glob.glob(os.path.join(args.raw_data, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in directory {args.raw_data}")
        csv_path = csv_files[0]
    else:
        csv_path = args.raw_data  # assume it's a direct CSV file path

    print(f"Reading CSV file from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Label encoding for object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Create output directories
    Path(args.train_data).mkdir(parents=True, exist_ok=True)
    Path(args.test_data).mkdir(parents=True, exist_ok=True)

    # Save CSVs
    train_path = Path(args.train_data) / "train.csv"
    test_path = Path(args.test_data) / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved test dataset to: {test_path}")

    # Log metrics
    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train data output path: {args.train_data}")
    print(f"Test data output path: {args.test_data}")
    print(f"Test/train ratio: {args.test_train_ratio}")

    main(args)
    mlflow.end_run()
