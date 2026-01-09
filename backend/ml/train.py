import argparse
import os
import numpy as np
import pandas as pd
from models import AnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="Train IsolationForest anomaly detector and save artifacts")
    parser.add_argument("--data-csv", type=str, default=None, help="Optional CSV file of normal training data")
    parser.add_argument("--artifacts", type=str, default=os.path.join(os.path.dirname(__file__), "artifacts"), help="Directory to save artifacts")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of simulated normal samples (used if no CSV provided)")
    args = parser.parse_args()

    detector = AnomalyDetector(artifacts_path=args.artifacts)

    if args.data_csv is not None:
        if not os.path.exists(args.data_csv):
            raise FileNotFoundError(f"CSV file not found: {args.data_csv}")
        df = pd.read_csv(args.data_csv)
        data = df.values
    else:
        data = None

    # If data is provided, pass it; otherwise detector will generate simulated normal data
    detector.train_model(normal_data=data, n_samples=args.n_samples)

    print(f"Training complete. Artifacts saved in: {args.artifacts}")


if __name__ == "__main__":
    main()
