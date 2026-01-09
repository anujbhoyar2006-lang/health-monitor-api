# Google Colab Setup Instructions

## Quick Start (Copy-Paste Ready)

### Step 1: Install Dependencies and Setup
```python
# Run this cell first - installs all dependencies and sets up directories
!pip install fastapi uvicorn[standard] pydantic scikit-learn tensorflow pandas numpy joblib python-multipart pyngrok

import os
import sys

# Create directory structure
os.makedirs("/content/backend", exist_ok=True)
os.makedirs("/content/backend/ml", exist_ok=True) 
os.makedirs("/content/backend/artifacts", exist_ok=True)

# Create __init__.py files
with open("/content/backend/__init__.py", "w") as f:
    f.write("# Backend package\n")
    
with open("/content/backend/ml/__init__.py", "w") as f:
    f.write("# ML package\n")

print("✅ Setup complete!")
```

### Step 2: Create the Preprocessing Module
```python
# Create preprocess.py
preprocess_code = '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, Any, Tuple

class DataPreprocessor:
    def __init__(self, scaler_path: str = "/content/backend/artifacts/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = None
        self.feature_columns = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature', 'glucose']
        
    def load_scaler(self):
        """Load the trained scaler from file"""
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            # Create and fit a new scaler with default normal ranges
            self.scaler = StandardScaler()
            # Default normal ranges for training the scaler
            normal_data = pd.DataFrame({
                'heart_rate': np.random.normal(75, 10, 1000),
                'respiratory_rate': np.random.normal(16, 3, 1000),
                'spo2': np.random.normal(98, 1.5, 1000),
                'temperature': np.random.normal(98.6, 0.8, 1000),
                'glucose': np.random.normal(100, 15, 1000)
            })
            self.scaler.fit(normal_data)
            self.save_scaler()
    
    def save_scaler(self):
        """Save the scaler to file"""
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def preprocess_vitals(self, vitals_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Preprocess incoming vitals data"""
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
        """Validate vitals data and return any errors"""
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
                    pass
        
        return errors
'''

with open("/content/backend/ml/preprocess.py", "w") as f:
    f.write(preprocess_code)

print("✅ Preprocessing module created!")
```

### Step 3: Create the ML Models Module
```python
# Create models.py
models_code = '''
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

class AnomalyDetector:
    def __init__(self, artifacts_path: str = "/content/backend/artifacts"):
        self.artifacts_path = artifacts_path
        self.isolation_forest = None
        self.autoencoder = None
        self.scaler = None
        
        # Ensure artifacts directory exists
        os.makedirs(artifacts_path, exist_ok=True)
        
    def create_autoencoder(self, input_dim: int = 5) -> keras.Model:
        """Create a simple autoencoder model for anomaly detection"""
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
        """Train both Isolation Forest and Autoencoder models"""
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
        """Generate synthetic normal vitals data for training"""
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
        """Predict anomaly using both models"""
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
'''

with open("/content/backend/ml/models.py", "w") as f:
    f.write(models_code)

print("✅ ML models module created!")
```

### Step 4: Create the FastAPI Main Application
```python
# Create main.py
main_code = '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
from datetime import datetime
import os
import sys

# Add the backend directory to Python path for imports
sys.path.append('/content')

from backend.ml.preprocess import DataPreprocessor
from backend.ml.models import AnomalyDetector

# Initialize FastAPI app
app = FastAPI(
    title="Remote Health Monitoring API",
    description="Real-time vitals monitoring with anomaly detection",
    version="1.0.0"
)

# Initialize components
preprocessor = DataPreprocessor()
anomaly_detector = AnomalyDetector()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and preprocessor on startup"""
    print("Loading models and preprocessor...")
    preprocessor.load_scaler()
    anomaly_detector.load_models()
    print("Startup complete!")

# Pydantic models for request/response
class VitalsRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    timestamp: str = Field(..., description="Timestamp in ISO format")
    heart_rate: Optional[float] = Field(None, description="Heart rate in BPM")
    respiratory_rate: Optional[float] = Field(None, description="Respiratory rate per minute")
    spo2: Optional[float] = Field(None, description="Blood oxygen saturation percentage")
    temperature: Optional[float] = Field(None, description="Body temperature in Fahrenheit")
    glucose: Optional[float] = Field(None, description="Blood glucose level in mg/dL")

class VitalsResponse(BaseModel):
    patient_id: str
    timestamp: str
    vitals: Dict[str, float]
    anomaly_score: float
    anomaly_detected: bool
    risk_level: str
    details: Dict[str, Any]
    processing_timestamp: str

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    models_loaded: bool

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = (
        anomaly_detector.isolation_forest is not None and 
        anomaly_detector.autoencoder is not None and
        preprocessor.scaler is not None
    )
    
    return HealthResponse(
        status="healthy" if models_loaded else "initializing",
        message="Remote Health Monitoring API is running",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded
    )

# Main vitals processing endpoint
@app.post("/analyze-vitals", response_model=VitalsResponse)
async def analyze_vitals(vitals: VitalsRequest):
    """Analyze patient vitals for anomalies"""
    try:
        # Convert request to dictionary
        vitals_data = vitals.dict()
        
        # Validate input data
        validation_errors = preprocessor.validate_vitals(vitals_data)
        if validation_errors:
            raise HTTPException(
                status_code=400, 
                detail=f"Validation errors: {validation_errors}"
            )
        
        # Preprocess the vitals data
        scaled_features, processed_vitals = preprocessor.preprocess_vitals(vitals_data)
        
        # Perform anomaly detection
        anomaly_results = anomaly_detector.predict_anomaly(scaled_features)
        
        # Prepare response
        response = VitalsResponse(
            patient_id=vitals.patient_id,
            timestamp=vitals.timestamp,
            vitals=processed_vitals,
            anomaly_score=anomaly_results['anomaly_score'],
            anomaly_detected=anomaly_results['anomaly_detected'],
            risk_level=anomaly_results['risk_level'],
            details={
                'isolation_forest_score': anomaly_results['isolation_forest_score'],
                'autoencoder_mse': anomaly_results['autoencoder_mse'],
                'isolation_forest_prediction': anomaly_results['isolation_forest_prediction']
            },
            processing_timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing vitals: {str(e)}"
        )

# Model information endpoint
@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "isolation_forest_loaded": anomaly_detector.isolation_forest is not None,
        "autoencoder_loaded": anomaly_detector.autoencoder is not None,
        "scaler_loaded": preprocessor.scaler is not None,
        "artifacts_path": anomaly_detector.artifacts_path,
        "feature_columns": preprocessor.feature_columns
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
'''

with open("/content/backend/main.py", "w") as f:
    f.write(main_code)

print("✅ FastAPI application created!")
```

### Step 5: Start the Server with ngrok
```python
# Start the server with public URL
import threading
import time
from pyngrok import ngrok
import uvicorn
import sys

# Add backend to Python path
sys.path.append('/content')

# Set up ngrok tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Public URL: {public_url}")
print(f"📋 API Documentation: {public_url}/docs")
print(f"🔍 Health Check: {public_url}/")

# Start FastAPI server in a separate thread
def start_server():
    from backend.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

print("🚀 Server starting...")
time.sleep(10)  # Give server time to start
print("✅ Server is running!")
print("\n" + "="*60)
print("API ENDPOINTS:")
print(f"• Health Check: GET {public_url}/")
print(f"• Analyze Vitals: POST {public_url}/analyze-vitals")
print(f"• Model Info: GET {public_url}/model-info")
print(f"• Interactive Docs: {public_url}/docs")
print("="*60)
```

### Step 6: Test the API
```python
# Test the API with sample data
import requests
import json

# Test health check
response = requests.get(f"{public_url}/")
print("Health Check Response:")
print(json.dumps(response.json(), indent=2))

# Test vitals analysis
test_vitals = {
    "patient_id": "P001",
    "timestamp": "2024-01-15T10:30:00Z",
    "heart_rate": 85,
    "respiratory_rate": 18,
    "spo2": 97,
    "temperature": 98.6,
    "glucose": 110
}

response = requests.post(f"{public_url}/analyze-vitals", json=test_vitals)
print("\nVitals Analysis Response:")
print(json.dumps(response.json(), indent=2))

# Test with anomalous data
anomalous_vitals = {
    "patient_id": "P002", 
    "timestamp": "2024-01-15T10:35:00Z",
    "heart_rate": 150,  # High heart rate
    "respiratory_rate": 35,  # High respiratory rate
    "spo2": 85,  # Low oxygen
    "temperature": 102.5,  # High fever
    "glucose": 250  # High glucose
}

response = requests.post(f"{public_url}/analyze-vitals", json=anomalous_vitals)
print("\nAnomalous Vitals Analysis Response:")
print(json.dumps(response.json(), indent=2))
```

## API Usage Examples

### Normal Vitals
```json
{
  "patient_id": "P001",
  "timestamp": "2024-01-15T10:30:00Z", 
  "heart_rate": 75,
  "respiratory_rate": 16,
  "spo2": 98,
  "temperature": 98.6,
  "glucose": 100
}
```

### Anomalous Vitals
```json
{
  "patient_id": "P002",
  "timestamp": "2024-01-15T10:35:00Z",
  "heart_rate": 150,
  "respiratory_rate": 35, 
  "spo2": 85,
  "temperature": 102.5,
  "glucose": 250
}
```

## Features

✅ **Dual Anomaly Detection**: Isolation Forest + Autoencoder  
✅ **Data Preprocessing**: Automatic scaling and missing value handling  
✅ **Risk Classification**: NORMAL, MODERATE, HIGH, CRITICAL levels  
✅ **Real-time Processing**: Fast API responses  
✅ **Public Access**: ngrok tunnel for external access  
✅ **Interactive Docs**: Automatic Swagger UI at `/docs`  
✅ **Health Monitoring**: Built-in health check endpoint  
✅ **Colab Ready**: All paths optimized for Google Colab  

The backend is now ready to receive real-time vitals data and provide intelligent anomaly detection!
