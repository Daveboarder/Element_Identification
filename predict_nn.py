"""
Element Identification Prediction Script (Classification Model)

This script:
1. Loads spectra from an input JSON file (reads both CCD ranges via get_spectra)
2. Loads multi-weight spectra from element_weights/multi_weights_vacuum.h5
3. Performs matrix multiplication: spectra @ weights.T  -> (n_samples, 7700)
4. Loads trained classification model (per-element branches, 1000 bins each)
5. Predicts concentration bin for each element, converts to concentration %

Prerequisite:
    Run weight_generator.py first, then train_nn_classification.py.

Usage:
    python predict_nn.py <input_json_file>
    python predict_nn.py <input_json_file> --output results.csv
    python predict_nn.py <input_json_file> --threshold 0.1
"""

import numpy as np
import h5py
import os
import pickle
import sys
from Sample_bootstrap import unit_norm
from LIBSmethods import movingMinimum
from readData import get_spectra, json_from_file, get_number_of_runs, is_run_valid

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
MULTI_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'element_weights', 'multi_weights_vacuum.h5')
MODEL_PT_PATH = os.path.join(SCRIPT_DIR, 'element_classification_model.pt')
MODEL_PKL_PATH = os.path.join(SCRIPT_DIR, 'element_classification_model.pkl')
N_BINS = 1000
DEFAULT_THRESHOLD = 0.1  # concentration fraction; elements above this are "detected"


# ============================================================================
# Bin Conversion Utilities
# ============================================================================

def bin_to_concentration(bin_idx):
    """Convert bin index [0, N_BINS-1] to concentration fraction [0.0, 1.0]."""
    return np.asarray(bin_idx, dtype=np.float64) / (N_BINS - 1)


# ============================================================================
# Neural Network Definitions (for inference)
# ============================================================================

def relu(x):
    return np.maximum(0, x)

def np_softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class NumpyBranchNN:
    """NumPy classification NN with per-element branches for inference."""

    def __init__(self, W0, W1, b1, branch_W1, branch_b1, branch_W2, branch_b2):
        self.W0 = W0
        self.W1, self.b1 = W1, b1
        self.branch_W1 = branch_W1
        self.branch_b1 = branch_b1
        self.branch_W2 = branch_W2
        self.branch_b2 = branch_b2
        self.n_elements = len(branch_W1)

    def predict(self, X):
        """Returns (n_samples, n_elements, n_bins) logits."""
        P = X @ self.W0
        Z1 = P @ self.W1 + self.b1
        A1 = relu(Z1)

        branch_outputs = []
        for e in range(self.n_elements):
            bz1 = A1 @ self.branch_W1[e] + self.branch_b1[e]
            ba1 = relu(bz1)
            logits = ba1 @ self.branch_W2[e] + self.branch_b2[e]
            branch_outputs.append(logits)

        return np.stack(branch_outputs, axis=1)


if HAS_TORCH:
    class ElementBranchNN(nn.Module):
        """PyTorch classification NN with per-element branches, matching train_nn_classification.py."""

        def __init__(self, n_features, n_elements, n_hidden, branch_hidden,
                     n_bins=N_BINS, dropout=0.0):
            super().__init__()
            self.n_elements = n_elements
            self.n_bins = n_bins

            self.projection = nn.Linear(n_features, n_elements, bias=False)
            self.shared = nn.Sequential(
                nn.Linear(n_elements, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_hidden, branch_hidden),
                    nn.ReLU(),
                    nn.Linear(branch_hidden, n_bins),
                )
                for _ in range(n_elements)
            ])

        def forward(self, x):
            x = self.projection(x)
            shared = self.shared(x)
            return torch.stack([branch(shared) for branch in self.branches], dim=1)


# ============================================================================
# Data Loading
# ============================================================================

def load_multi_weights():
    """Load multi-weight spectra from element_weights/multi_weights_vacuum.h5"""
    if not os.path.exists(MULTI_WEIGHTS_PATH):
        print(f"\nERROR: Multi-weight file not found: {MULTI_WEIGHTS_PATH}")
        print("Please run weight_generator.py first to generate the weight spectra.")
        sys.exit(1)

    with h5py.File(MULTI_WEIGHTS_PATH, 'r') as f:
        weight_matrix = f['weight_matrix'][:]
        unique_elements = [e.decode('utf-8') for e in f['unique_elements'][:]]

    return weight_matrix, unique_elements


def load_input_spectra(json_path):
    """
    Load spectra from an input JSON file.

    For each valid run, reads both CCD ranges via get_spectra (same approach
    as Sample_bootstrap.py lines 42-46), concatenates them, and applies
    unit_norm + movingMinimum normalization.

    Returns (spectra, n_valid_runs) where spectra is (n_runs, n_wavelengths).
    """
    if not os.path.exists(json_path):
        print(f"\nERROR: File not found: {json_path}")
        print("  Check the file path (including uppercase/lowercase).")
        sys.exit(1)

    j = json_from_file(json_path)['analysis']
    n_runs = get_number_of_runs(j)

    spectra_list = []
    for run_id in range(1, n_runs + 1):
        if not is_run_valid(j, run_id):
            continue
        try:
            w1, spectraData_1 = get_spectra(json_path, run_id, 1, 1)
            w2, spectraData_2 = get_spectra(json_path, run_id, 1, 2)
            spectrum = np.concatenate([spectraData_1, spectraData_2])
            spectrum = unit_norm(spectrum)
            spectrum = movingMinimum(spectrum)
            spectra_list.append(spectrum)
        except Exception as e:
            print(f"  Warning: Skipping run {run_id}: {e}")

    if not spectra_list:
        print(f"\nERROR: No valid spectra found in {json_path}")
        sys.exit(1)

    spectra = np.array(spectra_list, dtype=np.float32)
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    return spectra, len(spectra_list)


def load_model():
    """
    Load trained classification model. Tries PyTorch (.pt) first, then NumPy (.pkl).

    Returns (predict_fn, feature_mean, feature_std, weight_elements)
    where predict_fn(X) returns (n_samples, n_elements, n_bins) logits.
    """
    # Try PyTorch model first
    if HAS_TORCH and os.path.exists(MODEL_PT_PATH):
        print(f"  Loading PyTorch model: {MODEL_PT_PATH}")
        checkpoint = torch.load(MODEL_PT_PATH, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        model = ElementBranchNN(
            cfg['n_features'], cfg['n_elements'], cfg['n_hidden'],
            cfg['branch_hidden'], cfg.get('n_bins', N_BINS),
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

        model = NumpyBranchNN(
            data['W0'], data['W1'], data['b1'],
            data['branch_W1'], data['branch_b1'],
            data['branch_W2'], data['branch_b2'],
        )
        return (model.predict,
                data['feature_mean'],
                data['feature_std'],
                data['weight_elements'])

    print(f"\nERROR: No trained model found!")
    print(f"  Looked for: {MODEL_PT_PATH}")
    print(f"           or {MODEL_PKL_PATH}")
    print("Please run train_nn_classification.py first.")
    sys.exit(1)


# ============================================================================
# Prediction
# ============================================================================

def predict_elements(spectra, weight_matrix, predict_fn, feature_mean, feature_std):
    """
    Predict element concentrations for input spectra.

    Returns:
        concentrations: (n_samples, n_elements) predicted concentration fractions [0, 1]
        pred_bins:      (n_samples, n_elements) predicted bin indices [0, N_BINS-1]
    """
    features = spectra @ weight_matrix.T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    std_safe = feature_std.copy()
    std_safe[std_safe == 0] = 1
    features_norm = (features - feature_mean) / std_safe
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

    logits = predict_fn(features_norm)           # (n_samples, n_elements, n_bins)
    pred_bins = np.argmax(logits, axis=2)         # (n_samples, n_elements)
    concentrations = bin_to_concentration(pred_bins)  # (n_samples, n_elements) in [0, 1]

    return concentrations, pred_bins


# ============================================================================
# Output
# ============================================================================

def print_spectrum_prediction(conc, element_names, spectrum_idx, threshold=0.1):
    """Print full prediction table for a single spectrum."""
    print(f"\n{'='*70}")
    print(f"SPECTRUM #{spectrum_idx + 1} - ALL {len(element_names)} ELEMENTS")
    print(f"{'='*70}")

    detected = [
        (element_names[i], conc[i])
        for i in range(len(conc)) if conc[i] > threshold
    ]
    detected.sort(key=lambda x: x[1], reverse=True)

    print(f"\nDetected elements (threshold={threshold*100:.1f}%): {len(detected)}")
    if detected:
        print(f"  {[(e, f'{c*100:.1f}%') for e, c in detected]}")

    sorted_idx = np.argsort(conc)[::-1]
    print(f"\n{'Element':<8} {'Conc %':<12} {'Status'}")
    print("-" * 35)
    for idx in sorted_idx:
        status = "PRESENT" if conc[idx] > threshold else "absent"
        print(f"{element_names[idx]:<8} {conc[idx]*100:<12.2f} {status}")


def save_to_csv(concentrations, element_names, output_path):
    """Save predicted concentrations (%) to CSV file."""
    with open(output_path, 'w') as f:
        f.write('Spectrum_ID,' + ','.join(element_names) + '\n')
        for i in range(concentrations.shape[0]):
            vals = ','.join([f'{c*100:.4f}' for c in concentrations[i]])
            f.write(f'Spectrum_{i+1},{vals}\n')
    print(f"\nSaved predictions to: {output_path}")
    print(f"  {concentrations.shape[0]} spectra x {concentrations.shape[1]} elements (concentration %)")


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        print("Usage: python predict_nn.py <input_json_file> [--output file.csv] [--threshold 0.1]")
        print("\nArguments:")
        print("  input_json_file   Path to JSON file containing spectra (required)")
        print("  --output, -o      Output CSV file path (optional)")
        print("  --threshold, -t   Detection threshold as fraction 0-1 (default: 0.1 = 10%)")
        print("\nExample:")
        print("  python predict_nn.py measurement.json")
        print("  python predict_nn.py measurement.json --output results.csv --threshold 0.05")
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
    print("Element Identification - Classification Model Prediction")
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

    # 3. Load input spectra from JSON
    print(f"\nLoading input spectra from: {input_file}")
    input_spectra, n_valid_runs = load_input_spectra(input_file)
    print(f"  Valid runs: {n_valid_runs}")
    print(f"  Shape: {input_spectra.shape}")

    if input_spectra.shape[1] != weight_matrix.shape[1]:
        print(f"\nERROR: Wavelength mismatch!")
        print(f"  Input: {input_spectra.shape[1]} wavelengths")
        print(f"  Weights: {weight_matrix.shape[1]} wavelengths")
        print("Regenerate weights (weight_generator.py) with matching wavelength calibration.")
        sys.exit(1)

    # 4. Predict
    print(f"\nComputing predictions...")
    print(f"  Features: {input_spectra.shape[0]} spectra x {weight_matrix.shape[0]} weights")
    concentrations, pred_bins = predict_elements(
        input_spectra, weight_matrix, predict_fn, feature_mean, feature_std
    )
    print(f"  Predictions: {concentrations.shape}")

    # 5. Output summary
    print(f"\n{'='*70}")
    print(f"RESULTS - {concentrations.shape[0]} spectra, threshold={threshold*100:.1f}%")
    print(f"{'='*70}")

    for i in range(concentrations.shape[0]):
        conc = concentrations[i]
        detected = [
            (weight_elements[j], conc[j])
            for j in range(len(conc)) if conc[j] > threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)
        top_idx = np.argsort(conc)[::-1][:5]
        top_str = ', '.join([
            f"{weight_elements[j]}:{conc[j]*100:.1f}%" for j in top_idx
        ])
        print(f"\nSpectrum #{i+1}:")
        print(f"  Detected ({len(detected)}): "
              f"{[(e, f'{c*100:.1f}%') for e, c in detected] if detected else 'None'}")
        print(f"  Top 5: {top_str}")

    if output_file:
        save_to_csv(concentrations, weight_elements, output_file)

    print_spectrum_prediction(concentrations[0], weight_elements, 0, threshold)

    print("\n" + "=" * 70)
    print("Prediction Complete!")
    print("=" * 70)

    return concentrations, weight_elements


if __name__ == "__main__":
    concentrations, elements = main()
