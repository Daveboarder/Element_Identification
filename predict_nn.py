"""
Element Identification Prediction Script

This script:
1. Loads spectra from an input H5 file
2. Loads multi-weight spectra from element_weights/multi_weights.h5
3. Performs matrix multiplication: spectra @ weights.T  -> (n_samples, 7700)
4. Loads trained model (includes projection 7700->77 + hidden + output)
5. Predicts element presence (0-1) for all elements

Prerequisite:
    Run weight_generator.py first, then train_nn.py.

Usage:
    python predict_nn.py <input_h5_file>
    python predict_nn.py <input_h5_file> --output results.csv
    python predict_nn.py <input_h5_file> --threshold 0.3
"""

import numpy as np
import h5py
import os
import pickle
import sys

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'element_weights', 'multi_weights.h5')
MODEL_PT_PATH = os.path.join(SCRIPT_DIR, 'element_identification_model.pt')
MODEL_PKL_PATH = os.path.join(SCRIPT_DIR, 'element_identification_model.pkl')
DEFAULT_THRESHOLD = 0.5


# ============================================================================
# Neural Network Definitions (for inference)
# ============================================================================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def relu(x):
    return np.maximum(0, x)


class NumpyNN:
    """NumPy NN with projection layer for inference."""
    def __init__(self, W0, W1, b1, W2, b2):
        self.W0 = W0  # projection: (n_features, n_elements)
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
    
    def predict(self, X):
        P = X @ self.W0                    # projection
        Z1 = P @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return sigmoid(Z2)


if HAS_TORCH:
    class ElementIdentificationNN(nn.Module):
        """PyTorch NN with projection layer, matching train_nn.py."""
        def __init__(self, n_features, n_elements, n_hidden, n_outputs, dropout=0.0):
            super().__init__()
            self.projection = nn.Linear(n_features, n_elements, bias=False)
            self.network = nn.Sequential(
                nn.Linear(n_elements, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_hidden, n_outputs),
                nn.Sigmoid()
            )
        def forward(self, x):
            x = self.projection(x)
            return self.network(x)


# ============================================================================
# Data Loading
# ============================================================================

def load_multi_weights():
    """Load multi-weight spectra from element_weights/multi_weights.h5"""
    if not os.path.exists(MULTI_WEIGHTS_PATH):
        print(f"\nERROR: Multi-weight file not found: {MULTI_WEIGHTS_PATH}")
        print("Please run weight_generator.py first to generate the weight spectra.")
        sys.exit(1)
    
    with h5py.File(MULTI_WEIGHTS_PATH, 'r') as f:
        weight_matrix = f['weight_matrix'][:]
        unique_elements = [e.decode('utf-8') for e in f['unique_elements'][:]]
    
    return weight_matrix, unique_elements


def load_input_spectra(h5_path):
    """Load spectra from input H5 file."""
    if not os.path.exists(h5_path):
        print(f"\nERROR: File not found: {h5_path}")
        print("  Check the file path (including uppercase/lowercase).")
        sys.exit(1)
    
    common_paths = [
        'measurements/Measurement_1/libs/data',
        'data', 'spectra', 'libs/data'
    ]
    
    with h5py.File(h5_path, 'r') as f:
        for path in common_paths:
            try:
                spectra = f[path][:]
                if spectra.ndim == 1:
                    spectra = spectra.reshape(1, -1)
                return spectra, path
            except KeyError:
                continue
        
        available = []
        def collect_paths(name, obj):
            if isinstance(obj, h5py.Dataset):
                available.append(f"{name}: shape={obj.shape}")
        f.visititems(collect_paths)
        raise ValueError(f"Could not find spectra in {h5_path}.\nAvailable:\n" + "\n".join(available))


def load_model():
    """
    Load trained model. Tries PyTorch (.pt) first, then NumPy (.pkl).
    
    Returns (predict_fn, feature_mean, feature_std, weight_elements)
    """
    # Try PyTorch model first
    if HAS_TORCH and os.path.exists(MODEL_PT_PATH):
        print(f"  Loading PyTorch model: {MODEL_PT_PATH}")
        checkpoint = torch.load(MODEL_PT_PATH, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        model = ElementIdentificationNN(
            cfg['n_features'], cfg['n_elements'], cfg['n_hidden'], cfg['n_outputs']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        def predict_fn(X):
            with torch.no_grad():
                return model(torch.FloatTensor(X)).numpy()
        
        return (predict_fn,
                checkpoint['feature_mean'],
                checkpoint['feature_std'],
                checkpoint['weight_elements'])
    
    # Fall back to NumPy model
    if os.path.exists(MODEL_PKL_PATH):
        print(f"  Loading NumPy model: {MODEL_PKL_PATH}")
        with open(MODEL_PKL_PATH, 'rb') as f:
            data = pickle.load(f)
        
        model = NumpyNN(data['W0'], data['W1'], data['b1'], data['W2'], data['b2'])
        return (model.predict,
                data['feature_mean'],
                data['feature_std'],
                data['weight_elements'])
    
    print(f"\nERROR: No trained model found!")
    print(f"  Looked for: {MODEL_PT_PATH}")
    print(f"           or {MODEL_PKL_PATH}")
    print("Please run train_nn.py first.")
    sys.exit(1)


# ============================================================================
# Prediction
# ============================================================================

def predict_elements(spectra, weight_matrix, predict_fn, feature_mean, feature_std):
    """Predict element presence for input spectra."""
    # Matrix multiplication
    features = spectra @ weight_matrix.T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize using training statistics
    std_safe = feature_std.copy()
    std_safe[std_safe == 0] = 1
    features_norm = (features - feature_mean) / std_safe
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    return predict_fn(features_norm)


# ============================================================================
# Output
# ============================================================================

def print_spectrum_prediction(pred, element_names, spectrum_idx, threshold=0.5):
    """Print full prediction table for a single spectrum."""
    print(f"\n{'='*70}")
    print(f"SPECTRUM #{spectrum_idx + 1} - ALL {len(element_names)} ELEMENTS")
    print(f"{'='*70}")
    
    detected = [(element_names[i], pred[i]) for i in range(len(pred)) if pred[i] > threshold]
    detected.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nDetected elements (threshold={threshold}): {len(detected)}")
    if detected:
        print(f"  {[e[0] for e in detected]}")
    
    sorted_idx = np.argsort(pred)[::-1]
    print(f"\n{'Element':<8} {'Probability':<12} {'Status'}")
    print("-" * 35)
    for idx in sorted_idx:
        status = "PRESENT" if pred[idx] > threshold else "absent"
        print(f"{element_names[idx]:<8} {pred[idx]:<12.4f} {status}")


def save_to_csv(predictions, element_names, output_path, threshold=0.5):
    """Save predictions to CSV file."""
    with open(output_path, 'w') as f:
        f.write('Spectrum_ID,' + ','.join(element_names) + ',Detected_Elements\n')
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            detected = [element_names[j] for j in range(len(pred)) if pred[j] > threshold]
            detected_str = ';'.join(detected) if detected else 'None'
            probs = ','.join([f'{p:.4f}' for p in pred])
            f.write(f'Spectrum_{i+1},{probs},{detected_str}\n')
    print(f"\nSaved predictions to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse arguments
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        print("Usage: python predict_nn.py <input_h5_file> [--output file.csv] [--threshold 0.5]")
        print("\nArguments:")
        print("  input_h5_file     Path to H5 file containing spectra (required)")
        print("  --output, -o      Output CSV file path (optional)")
        print("  --threshold, -t   Detection threshold 0-1 (default: 0.5)")
        print("\nExample:")
        print("  python predict_nn.py synthetic_data.h5")
        print("  python predict_nn.py measurement.h5 --output results.csv --threshold 0.3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    threshold = DEFAULT_THRESHOLD
    
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--output', '-o'] and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg in ['--threshold', '-t'] and i + 1 < len(sys.argv):
            try:
                threshold = float(sys.argv[i + 1])
            except ValueError:
                print(f"Error: Invalid threshold '{sys.argv[i + 1]}'. Must be a number.")
                sys.exit(1)
            i += 2
        else:
            print(f"Warning: Unknown argument '{arg}', ignoring.")
            i += 1
    
    print("=" * 70)
    print("Element Identification - Prediction")
    print("=" * 70)
    
    # 1. Load multi-weight spectra
    print(f"\nLoading weight spectra from: {MULTI_WEIGHTS_PATH}")
    weight_matrix, weight_elements = load_multi_weights()
    print(f"  Weight matrix: {weight_matrix.shape}")
    print(f"  Elements: {len(weight_elements)}")
    
    # 2. Load model
    print(f"\nLoading trained model...")
    predict_fn, feature_mean, feature_std, model_elements = load_model()
    print("  Model loaded successfully")
    
    # 3. Load input spectra
    print(f"\nLoading input spectra from: {input_file}")
    input_spectra, data_path = load_input_spectra(input_file)
    print(f"  Found at: '{data_path}'")
    print(f"  Shape: {input_spectra.shape}")
    
    # Check wavelength compatibility
    if input_spectra.shape[1] != weight_matrix.shape[1]:
        print(f"\nERROR: Wavelength mismatch!")
        print(f"  Input: {input_spectra.shape[1]} wavelengths")
        print(f"  Weights: {weight_matrix.shape[1]} wavelengths")
        print("Regenerate weights (weight_generator.py) with matching wavelength calibration.")
        sys.exit(1)
    
    # 4. Predict
    print(f"\nComputing predictions...")
    print(f"  Features: {input_spectra.shape[0]} spectra x {weight_matrix.shape[0]} weights")
    predictions = predict_elements(input_spectra, weight_matrix, predict_fn, feature_mean, feature_std)
    print(f"  Predictions: {predictions.shape}")
    
    # 5. Output summary
    print(f"\n{'='*70}")
    print(f"RESULTS - {predictions.shape[0]} spectra, threshold={threshold}")
    print(f"{'='*70}")
    
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        detected = [weight_elements[j] for j in range(len(pred)) if pred[j] > threshold]
        top_idx = np.argsort(pred)[::-1][:5]
        top_str = ', '.join([f"{weight_elements[j]}:{pred[j]:.3f}" for j in top_idx])
        print(f"\nSpectrum #{i+1}:")
        print(f"  Detected ({len(detected)}): {detected if detected else 'None'}")
        print(f"  Top 5: {top_str}")
    
    # Save to CSV if requested
    if output_file:
        save_to_csv(predictions, weight_elements, output_file, threshold)
    
    # Full table for first spectrum
    print_spectrum_prediction(predictions[0], weight_elements, 0, threshold)
    
    print("\n" + "=" * 70)
    print("Prediction Complete!")
    print("=" * 70)
    
    return predictions, weight_elements


if __name__ == "__main__":
    predictions, elements = main()
