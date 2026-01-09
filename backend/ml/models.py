import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
from typing import Tuple, Dict, Any
import joblib

class AnomalyDetector:
    def __init__(self, artifacts_path: str = "/content/backend/artifacts"):
        self.artifacts_path = artifacts_path
        self.isolation_forest = None
        self.autoencoder = None
        self.scaler = None
        
        # Ensure artifacts directory exists
        os.makedirs(artifacts_path, exist_ok=True)
        
    def create_autoencoder(self, input_dim: int = 5) -> keras.Model:
        """
        Create a simple autoencoder model for anomaly detection
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled autoencoder model
        """
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(8, activation='relu')(input_layer)
        encoded = layers.Dense(4, activation='relu')(encoded)
        encoded = layers.Dense(2, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(4, activation='relu')(encoded)
        decoded = layers.Dense(8, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def train_models(self, normal_data: np.ndarray = None):
        """
        Train both Isolation Forest and Autoencoder models
        
        Args:
            normal_data: Training data (normal vitals only)
        """
        # Generate synthetic normal data if none provided
        if normal_data is None:
            normal_data = self._generate_normal_data()
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(normal_data)
        
        # Train Autoencoder
        self.autoencoder = self.create_autoencoder(input_dim=normal_data.shape[1])
        
        # Train autoencoder on normal data
        history = self.autoencoder.fit(
            normal_data, normal_data,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            shuffle=True
        )
        
        # Save models
        self.save_models()
        
        return history
    
    def _generate_normal_data(self, n_samples: int = 5000) -> np.ndarray:
        """
        Generate synthetic normal vitals data for training
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Normalized training data
        """
        # Generate normal vitals with realistic distributions
        np.random.seed(42)
        
        data = {
            'heart_rate': np.random.normal(75, 10, n_samples),
            'respiratory_rate': np.random.normal(16, 3, n_samples),
            'spo2': np.random.normal(98, 1.5, n_samples),
            'temperature': np.random.normal(98.6, 0.8, n_samples),
            'glucose': np.random.normal(100, 15, n_samples)
        }
        
        # Add some correlation between features for realism
        for i in range(n_samples):
            # Higher heart rate might correlate with slightly higher temperature
            if data['heart_rate'][i] > 85:
                data['temperature'][i] += np.random.normal(0.5, 0.2)
            
            # Ensure realistic ranges
            data['heart_rate'][i] = np.clip(data['heart_rate'][i], 50, 120)
            data['respiratory_rate'][i] = np.clip(data['respiratory_rate'][i], 12, 25)
            data['spo2'][i] = np.clip(data['spo2'][i], 95, 100)
            data['temperature'][i] = np.clip(data['temperature'][i], 97, 100)
            data['glucose'][i] = np.clip(data['glucose'][i], 70, 140)
        
        df = pd.DataFrame(data)
        
        # Normalize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df)
        
        # Save the scaler
        scaler_path = os.path.join(self.artifacts_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        return normalized_data
    
    def load_models(self):
        """Load trained models from artifacts directory"""
        # Load Isolation Forest
        if_path = os.path.join(self.artifacts_path, 'isolation_forest.pkl')
        if os.path.exists(if_path):
            with open(if_path, 'rb') as f:
                self.isolation_forest = pickle.load(f)
        
        # Load Autoencoder
        ae_path = os.path.join(self.artifacts_path, 'autoencoder.h5')
        if os.path.exists(ae_path):
            self.autoencoder = keras.models.load_model(ae_path)
        
        # If models don't exist, train them
        if self.isolation_forest is None or self.autoencoder is None:
            print("Models not found. Training new models...")
            self.train_models()
    
    def save_models(self):
        """Save trained models to artifacts directory"""
        # Save Isolation Forest
        if_path = os.path.join(self.artifacts_path, 'isolation_forest.pkl')
        with open(if_path, 'wb') as f:
            pickle.dump(self.isolation_forest, f)
        
        # Save Autoencoder
        ae_path = os.path.join(self.artifacts_path, 'autoencoder.h5')
        self.autoencoder.save(ae_path)
        
        print(f"Models saved to {self.artifacts_path}")
    
    def predict_anomaly(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict anomaly using both models
        
        Args:
            features: Preprocessed and scaled features
            
        Returns:
            Dictionary with anomaly predictions and scores
        """
        # Ensure models are loaded
        if self.isolation_forest is None or self.autoencoder is None:
            self.load_models()
        
        # Isolation Forest prediction
        if_prediction = self.isolation_forest.predict(features)[0]
        if_score = self.isolation_forest.decision_function(features)[0]
        
        # Autoencoder prediction
        ae_reconstruction = self.autoencoder.predict(features, verbose=0)
        ae_mse = np.mean(np.square(features - ae_reconstruction))
        
        # Combine scores (weighted average)
        # Normalize IF score to 0-1 range (higher = more anomalous)
        if_score_normalized = max(0, (0.5 - if_score) / 0.5)
        
        # Normalize AE score (MSE) - typical normal MSE is around 0.01-0.1
        ae_score_normalized = min(1.0, ae_mse / 0.1)
        
        # Combined anomaly score
        combined_score = (if_score_normalized * 0.6) + (ae_score_normalized * 0.4)
        
        # Determine if anomaly detected
        anomaly_detected = if_prediction == -1 or ae_mse > 0.05
        
        # Determine risk level based on combined score
        if combined_score < 0.3:
            risk_level = "NORMAL"
        elif combined_score < 0.6:
            risk_level = "MODERATE"
        elif combined_score < 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            'anomaly_score': float(combined_score),
            'anomaly_detected': bool(anomaly_detected),
            'risk_level': risk_level,
            'isolation_forest_score': float(if_score_normalized),
            'autoencoder_mse': float(ae_mse),
            'isolation_forest_prediction': int(if_prediction)
        }
