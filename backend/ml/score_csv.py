import argparse
import pandas as pd
import os
from models import AnomalyDetector


def main():
    parser = argparse.ArgumentParser(description="Score a CSV of vitals with the trained IsolationForest")
    parser.add_argument("input_csv", type=str, help="Input CSV with columns in order: heart_rate,respiratory_rate,spo2,temperature,glucose")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path (defaults to input.scored.csv)")
    parser.add_argument("--artifacts", type=str, default=os.path.join(os.path.dirname(__file__), "artifacts"), help="Artifacts directory")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 'glucose']
    if not all(c in df.columns for c in required):
        raise ValueError(f"Input CSV must contain columns: {required}")

    det = AnomalyDetector(artifacts_path=args.artifacts)
    det.load_models()

    X = df[required].values
    results = det.predict_anomaly(X)

    if isinstance(results, dict):
        # single row
        results = [results]

    # expand results into columns
    df_scores = pd.DataFrame(results)
    # scaled features is a list; expand into columns
    scaled_cols = [f"scaled_{c}" for c in required]
    scaled_df = pd.DataFrame(df_scores.pop('scaled_features').tolist(), columns=scaled_cols)

    out = pd.concat([df.reset_index(drop=True), df_scores.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)

    out_path = args.output_csv or (os.path.splitext(args.input_csv)[0] + ".scored.csv")
    out.to_csv(out_path, index=False)
    print(f"Scored CSV written to: {out_path}")


if __name__ == '__main__':
    main()
