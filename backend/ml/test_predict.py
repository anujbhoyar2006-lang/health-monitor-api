from backend.ml.models import AnomalyDetector
import numpy as np

det = AnomalyDetector('backend/ml/artifacts')
det.load_models()
print('Loaded OK')
print('Single:')
print(det.predict_anomaly(np.array([80,18,97,98.6,110])))
print('Batch:')
print(det.predict_anomaly(np.array([[80,18,97,98.6,110],[40,10,88,100,60]])))
