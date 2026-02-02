import numpy as np
import pickle

import numpy as np

class IsotonicCalibrator:
    """Wrapper for calibrated probabilities using Isotonic Regression."""
    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        # Base estimator returns (N, 2), we take proba of class 1
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        # Clip to ensure valid probabilities
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

def load_calibrator(path):
    """Robustly load a calibrator pickle, handling module path changes."""
    # Shim to redirect old script module path to new location
    # This allows unpickling objects that were saved as 'scripts.route_xgb_results.IsotonicCalibrator'
    # or '__main__.IsotonicCalibrator'
    
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'IsotonicCalibrator':
                return IsotonicCalibrator
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return Unpickler(f).load()
