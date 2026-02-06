"""
Element Identification Prediction Script

This script:
1. Loads spectra from an input H5 file
2. Loads weight spectra from element_weights.h5
3. Performs matrix multiplication: spectra @ weights.T
4. Loads the trained model and predicts element presence (0-1) for all 77 elements

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

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'element_weights', 'element_weights.h5')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'element_identification_model.pkl')

# Default threshold for element detection
DEFAULT_THRESHOLD = 0.5


# ============================================================================
# Neural Network Functions
# ============================================================================

def sigmoid(x):
    """Sigmoid activation with numerical stability."""
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


class NumpyNN:
    """NumPy neural network for inference."""
    
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
    
    def predict(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return sigmoid(Z2)


# ============================================================================
# Data Loading
# ============================================================================

def load_weight_spectra():
    """Load weight spectra from element_weights.h5"""
    with h5py.File(WEIGHTS_PATH, 'r') as f:
        weight_spectra = f['measurements/Measurement_1/libs/data'][:]
        weight_elements = [e.decode('utf-8') for e in f['measurements/Measurement_1/libs/metadata/elements'][:]]
    return weight_spectra, weight_elements


def load_input_spectra(h5_path):
    """Load spectra from input H5 file."""
    # Check if file exists first
    if not os.path.exists(h5_path):
        print(f"\nERROR: File not found: {h5_path}")
        print("\nPlease check:")
        print("  1. The file path is correct (check uppercase/lowercase)")
        print("  2. The file exists")
        sys.exit(1)
    
    common_paths = [
        'measurements/Measurement_1/libs/data',
        'data',
        'spectra',
        'libs/data'
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
        
        # If not found, raise error with available paths
        available = []
        def collect_paths(name, obj):
            if isinstance(obj, h5py.Dataset):
                available.append(f"{name}: shape={obj.shape}")
        f.visititems(collect_paths)
        
        raise ValueError(f"Could not find spectra in {h5_path}.\nAvailable datasets:\n" + "\n".join(available))


def load_model():
    """Load trained model."""
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    
    model = NumpyNN(data['W1'], data['b1'], data['W2'], data['b2'])
    return model, data['feature_mean'], data['feature_std'], data['weight_elements']


# ============================================================================
# Prediction
# ============================================================================

def predict_elements(spectra, weight_spectra, model, feature_mean, feature_std):
    """
    Predict element presence for input spectra.
    
    Returns predictions of shape (n_samples, 77) with values 0-1.
    """
    # Matrix multiplication: spectra @ weights.T
    features = spectra @ weight_spectra.T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize
    std_safe = feature_std.copy()
    std_safe[std_safe == 0] = 1
    features_norm = (features - feature_mean) / std_safe
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Predict
    predictions = model.predict(features_norm)
    
    return predictions


# ============================================================================
# Output
# ============================================================================

def print_spectrum_prediction(pred, element_names, spectrum_idx, threshold=0.5):
    """Print prediction for a single spectrum."""
    print(f"\n{'='*70}")
    print(f"SPECTRUM #{spectrum_idx + 1} - ALL 77 ELEMENTS")
    print(f"{'='*70}")
    
    # Detected elements
    detected = [(element_names[i], pred[i]) for i in range(len(pred)) if pred[i] > threshold]
    detected.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nDetected elements (threshold={threshold}): {len(detected)}")
    if detected:
        print(f"  {[e[0] for e in detected]}")
    
    # Full table sorted by probability
    sorted_idx = np.argsort(pred)[::-1]
    
    print(f"\n{'Element':<8} {'Probability':<12} {'Status'}")
    print("-" * 35)
    
    for idx in sorted_idx:
        elem = element_names[idx]
        prob = pred[idx]
        status = "PRESENT" if prob > threshold else "absent"
        print(f"{elem:<8} {prob:<12.4f} {status}")


def save_to_csv(predictions, element_names, output_path, threshold=0.5):
    """Save predictions to CSV file."""
    with open(output_path, 'w') as f:
        # Header
        f.write('Spectrum_ID,' + ','.join(element_names) + ',Detected_Elements\n')
        
        # Data
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
        print("  python predict_nn.py measurement.h5 --output results.csv")
        print("  python predict_nn.py data.h5 --threshold 0.3 --output predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    threshold = DEFAULT_THRESHOLD
    
    # Parse optional arguments
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
                print(f"Error: Invalid threshold value '{sys.argv[i + 1]}'. Must be a number.")
                sys.exit(1)
            i += 2
        else:
            print(f"Warning: Unknown argument '{arg}', ignoring.")
            i += 1
    
    print("=" * 70)
    print("Element Identification - Prediction")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading weight spectra from: {WEIGHTS_PATH}")
    weight_spectra, weight_elements = load_weight_spectra()
    print(f"  Shape: {weight_spectra.shape}, Elements: {len(weight_elements)}")
    
    print(f"\nLoading model from: {MODEL_PATH}")
    model, feature_mean, feature_std, _ = load_model()
    print("  Model loaded successfully")
    
    print(f"\nLoading input spectra from: {input_file}")
    input_spectra, data_path = load_input_spectra(input_file)
    print(f"  Found at: '{data_path}'")
    print(f"  Shape: {input_spectra.shape}")
    
    # Check dimensions
    if input_spectra.shape[1] != weight_spectra.shape[1]:
        print(f"\nERROR: Wavelength mismatch!")
        print(f"  Input: {input_spectra.shape[1]} wavelengths")
        print(f"  Weights: {weight_spectra.shape[1]} wavelengths")
        sys.exit(1)
    
    # Predict
    print(f"\nComputing predictions...")
    print(f"  Matrix multiplication: {input_spectra.shape} @ {weight_spectra.T.shape}")
    predictions = predict_elements(input_spectra, weight_spectra, model, feature_mean, feature_std)
    print(f"  Predictions shape: {predictions.shape}")
    
    # Output
    print(f"\n{'='*70}")
    print(f"RESULTS - {predictions.shape[0]} spectra, threshold={threshold}")
    print(f"{'='*70}")
    
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        detected = [weight_elements[j] for j in range(len(pred)) if pred[j] > threshold]
        
        # Top 5 probabilities
        top_idx = np.argsort(pred)[::-1][:5]
        top_str = ', '.join([f"{weight_elements[j]}:{pred[j]:.3f}" for j in top_idx])
        
        print(f"\nSpectrum #{i+1}:")
        print(f"  Detected ({len(detected)}): {detected if detected else 'None'}")
        print(f"  Top 5: {top_str}")
    
    # Save to CSV if requested
    if output_file:
        save_to_csv(predictions, weight_elements, output_file, threshold)
    
    # Print full table for first spectrum
    print_spectrum_prediction(predictions[0], weight_elements, 0, threshold)
    
    print("\n" + "=" * 70)
    print("Prediction Complete!")
    print("=" * 70)
    
    return predictions, weight_elements


if __name__ == "__main__":
    predictions, elements = main()
