import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os
import logging
from typing import Dict, Any, List, Union

from backend.ml.preprocess import DataPreprocessor

logger = logging.getLogger(__name__) 


class AnomalyDetector:
    def __init__(self, artifacts_path: str = "artifacts"):
        self.artifacts_path = artifacts_path
        self.model = None
        self.preprocessor = DataPreprocessor(scaler_path=os.path.join(self.artifacts_path, "scaler.pkl"))

        os.makedirs(self.artifacts_path, exist_ok=True)

    def train_model(self, normal_data: Union[np.ndarray, pd.DataFrame] = None, n_samples: int = 5000):
        try:
            if normal_data is None:
                normal_data = self._generate_normal_data(n_samples=n_samples)

            # ensure dataframe and correct columns
            if isinstance(normal_data, np.ndarray):
                df = pd.DataFrame(normal_data, columns=["heart_rate", "respiratory_rate", "spo2", "temperature", "glucose"])
            else:
                df = normal_data

            # Fit scaler (preprocessor) on clean normal data
            self.preprocessor.fit_scaler(df)
            scaled_data = self.preprocessor.transform_array(df)

            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=42
            )
            self.model.fit(scaled_data)

            self.save_models()
            logger.info("Training completed and artifacts saved to %s", self.artifacts_path)
        except Exception:
            logger.exception("Training failed.")

    def _generate_normal_data(self, n_samples: int = 5000) -> np.ndarray:
        np.random.seed(42)

        data = {
            "heart_rate": np.random.normal(75, 10, n_samples),
            "respiratory_rate": np.random.normal(16, 3, n_samples),
            "spo2": np.random.normal(98, 1.5, n_samples),
            "temperature": np.random.normal(98.6, 0.8, n_samples),
            "glucose": np.random.normal(100, 15, n_samples),
        }

        df = pd.DataFrame(data)

        df["heart_rate"] = df["heart_rate"].clip(50, 120)
        df["respiratory_rate"] = df["respiratory_rate"].clip(12, 25)
        df["spo2"] = df["spo2"].clip(95, 100)
        df["temperature"] = df["temperature"].clip(97, 100)
        df["glucose"] = df["glucose"].clip(70, 140)

        return df

    def save_models(self):
        with open(os.path.join(self.artifacts_path, "isolation_forest.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        # save scaler via preprocessor
        self.preprocessor.save_scaler()

    def load_models(self):
        model_path = os.path.join(self.artifacts_path, "isolation_forest.pkl")

        try:
            if not os.path.exists(model_path):
                # Controlled auto-training on startup via env var
                auto_train = os.environ.get("MODEL_WARMUP", "true").lower() in ("1", "true", "yes")
                if not auto_train:
                    logger.warning("Model artifact missing and MODEL_WARMUP is disabled. Skipping training.")
                    return

                startup_samples = int(os.environ.get("STARTUP_N_SAMPLES", "500"))
                logger.info("Model artifact missing. Auto-training with %s samples (startup scale).", startup_samples)
                try:
                    # Use reduced samples for startup to avoid resource exhaustion
                    self.train_model(n_samples=startup_samples)
                except Exception:
                    logger.exception("Auto-training failed during startup; leaving model unloaded.")
                    return
            else:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)

            # ensure scaler is loaded; handle missing/corrupt scaler gracefully
            try:
                self.preprocessor.load_scaler()
            except Exception:
                logger.exception("Failed to load scaler; scaler will be refit on next training.")
        except Exception:
            logger.exception("Unexpected error while loading models; continuing without a loaded model.")
            # ensure we don't crash the app at startup
            return

    def predict_anomaly(self, features: Union[np.ndarray, List[float]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Accepts a single sample (1D array/list) or batch (2D numpy array).

        Returns:
            - dict for single sample
            - list of dicts for batch
        """
        if self.model is None or self.preprocessor is None:
            self.load_models()

        if self.model is None:
            raise RuntimeError("Model is not available. Trigger training via /train or enable MODEL_WARMUP for auto-training on startup")

        # Normalize input to array
        arr = np.asarray(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        scaled_features = self.preprocessor.transform_array(arr)

        predictions = self.model.predict(scaled_features)
        scores = self.model.decision_function(scaled_features)

        # Normalize scores to 0–1
        anomaly_scores = np.clip((0.5 - scores) / 0.5, 0, 1)

        results = []
        for pred, score, a_score, scaled in zip(predictions, scores, anomaly_scores, scaled_features):
            if a_score < 0.3:
                risk_level = "NORMAL"
            elif a_score < 0.6:
                risk_level = "MODERATE"
            elif a_score < 0.8:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            results.append({
                "anomaly_score": float(a_score),
                "anomaly_detected": bool(pred == -1),
                "risk_level": risk_level,
                "isolation_forest_score": float(a_score),
                "isolation_forest_prediction": int(pred),
                "scaled_features": [float(x) for x in scaled]
            })

        if len(results) == 1:
            return results[0]
        return results

