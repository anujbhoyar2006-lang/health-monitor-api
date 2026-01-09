import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, Any, Tuple

class DataPreprocessor:
    def __init__(self, scaler_path: str = None):
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(__file__), "artifacts", "scaler.pkl")
        self.scaler_path = scaler_path
        self.scaler = None
        self.feature_columns = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 'glucose']
        
    def load_scaler(self):
        """Load the trained scaler from file or create a default one if missing"""
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            # Create a default scaler using sane normal ranges
            self.scaler = StandardScaler()
            normal_data = pd.DataFrame({
                'heart_rate': np.random.normal(75, 10, 1000),
                'respiratory_rate': np.random.normal(16, 3, 1000),
                'spo2': np.random.normal(98, 1.5, 1000),
                'temperature': np.random.normal(98.6, 0.8, 1000),
                'glucose': np.random.normal(100, 15, 1000)
            })
            self.scaler.fit(normal_data)
            self.save_scaler()

    def fit_scaler(self, X):
        """Fit the scaler on provided data (numpy array or DataFrame) and save it"""
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.feature_columns)

        self.scaler = StandardScaler()
        self.scaler.fit(df[self.feature_columns])
        self.save_scaler()

    def transform_array(self, X):
        """Transform an array or DataFrame using the stored scaler"""
        if self.scaler is None:
            self.load_scaler()

        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.feature_columns)

        return self.scaler.transform(df[self.feature_columns])
    
    def save_scaler(self):
        """Save the scaler to file"""
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def preprocess_vitals(self, vitals_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess incoming vitals data
        
        Args:
            vitals_data: Dictionary containing vital signs data
            
        Returns:
            Tuple of (scaled_features, processed_vitals)
        """
        # Extract and convert to numeric
        processed_vitals = {}
        
        for feature in self.feature_columns:
            value = vitals_data.get(feature)
            
            # Convert to numeric, handle various input types
            if value is None or value == '' or value == 'null':
                processed_vitals[feature] = np.nan
            else:
                try:
                    processed_vitals[feature] = float(value)
                except (ValueError, TypeError):
                    processed_vitals[feature] = np.nan
        
        # Create DataFrame for easier handling
        df = pd.DataFrame([processed_vitals])
        
        # Fill missing values with median of normal ranges
        median_values = {
            'heart_rate': 75.0,
            'respiratory_rate': 16.0,
            'spo2': 98.0,
            'temperature': 98.6,
            'glucose': 100.0
        }
        
        for feature in self.feature_columns:
            if pd.isna(df[feature].iloc[0]):
                df[feature] = median_values[feature]
                processed_vitals[feature] = median_values[feature]
        
        # Ensure scaler is loaded
        if self.scaler is None:
            self.load_scaler()
        
        # Scale the features
        scaled_features = self.scaler.transform(df[self.feature_columns])
        
        return scaled_features, processed_vitals
    
    def validate_vitals(self, vitals_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate vitals data and return any errors
        
        Args:
            vitals_data: Dictionary containing vital signs data
            
        Returns:
            Dictionary of validation errors
        """
        errors = {}
        
        # Check for required fields
        required_fields = ['patient_id', 'timestamp'] + self.feature_columns
        for field in required_fields:
            if field not in vitals_data:
                errors[field] = f"{field} is required"
        
        # Validate ranges (after conversion to numeric)
        ranges = {
            'heart_rate': (30, 200),
            'respiratory_rate': (8, 40),
            'spo2': (70, 100),
            'temperature': (90, 110),
            'glucose': (20, 500)
        }
        
        for feature, (min_val, max_val) in ranges.items():
            if feature in vitals_data:
                try:
                    value = float(vitals_data[feature])
                    if not (min_val <= value <= max_val):
                        errors[feature] = f"{feature} must be between {min_val} and {max_val}"
                except (ValueError, TypeError):
                    # Will be handled in preprocessing
                    pass
        
        return errors
