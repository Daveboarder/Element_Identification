"""
Element Identification Prediction Script (Transformer Model)

This script:
1. Loads spectra from an input JSON file (reads both CCD ranges via get_spectra)
2. Loads trained Transformer model (element_transformer_model.pt)
3. Converts spectra to spectral tokens [intensity, wavelength]
4. Predicts concentration bin for each element, converts to concentration %

Prerequisite:
    Run train_nn_autotransformer.py first.

Usage:
    python predict_nn.py <input_json_file>
    python predict_nn.py <input_json_file> --output results.csv
    python predict_nn.py <input_json_file> --threshold 0.1
"""

import json
import numpy as np
import os
import sys
from Sample_bootstrap import unit_norm
from LIBSmethods import movingMinimum
from readData import get_spectra, json_from_file, get_number_of_runs, is_run_valid

import torch
import torch.nn as nn

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "element_transformer_test_v5"
MODEL_PATH = os.path.join(SCRIPT_DIR, "experiments", EXPERIMENT_NAME, "best_model.pt")
VALIDATION_PATH = os.path.join(SCRIPT_DIR, "experiments", EXPERIMENT_NAME, "validation")
N_BINS = 1000
DEFAULT_THRESHOLD = 0.1


# ============================================================================
# Bin Conversion Utilities
# ============================================================================

def bin_to_concentration(bin_idx):
    """Convert bin index [0, N_BINS-1] to concentration fraction [0.0, 1.0]."""
    return np.asarray(bin_idx, dtype=np.float64) / (N_BINS - 1)

# ============================================================================
# Neural Network Definitions (must match train_nn_autotransformer.py)
# ============================================================================

class SpectralEmbedding(nn.Module):
    """Project each 2D spectral token [intensity, wavelength] to d_model dimensions."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(2, d_model)

    def forward(self, x):
        return self.linear(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1)]


class SpectralTransformerNN(nn.Module):
    """
    Transformer encoder for spectral element classification.

    Input:  (batch, seq_len, 2) spectral tokens [intensity, wavelength]
    Output: (batch, n_elements, n_bins) logits per element
    """

    def __init__(self, d_model, n_heads, n_layers, dim_ff, n_elements,
                 branch_hidden, n_bins=N_BINS, max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.n_elements = n_elements
        self.n_bins = n_bins

        self.embedding = SpectralEmbedding(d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len + 100)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_norm = nn.LayerNorm(d_model)

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, branch_hidden),
                nn.ReLU(),
                nn.Linear(branch_hidden, n_bins),
            )
            for _ in range(n_elements)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding(x)
        x = self.embed_dropout(x)
        x = self.transformer(x)
        x = self.pool_norm(x.mean(dim=1))
        return torch.stack([branch(x) for branch in self.branches], dim=1)


# ============================================================================
# Data Loading
# ============================================================================

def load_input_spectra(json_path):
    """
    Load spectra and wavelengths from an input JSON file.

    For each valid run, reads both CCD ranges via get_spectra (same approach
    as Sample_bootstrap.py), concatenates them, and applies
    unit_norm + movingMinimum normalization.

    Returns (spectra, wavelength, n_valid_runs) where
        spectra is (n_runs, n_wavelengths) and
        wavelength is (n_wavelengths,).
    """
    if not os.path.exists(json_path):
        print(f"\nERROR: File not found: {json_path}")
        print("  Check the file path (including uppercase/lowercase).")
        sys.exit(1)

    j = json_from_file(json_path)['analysis']
    n_runs = get_number_of_runs(j)

    spectra_list = []
    wavelength = None
    for run_id in range(1, n_runs + 1):
        if not is_run_valid(j, run_id):
            continue
        try:
            w1, spectraData_1 = get_spectra(json_path, run_id, 1, 1)
            w2, spectraData_2 = get_spectra(json_path, run_id, 1, 2)
            spectrum = np.concatenate([spectraData_1, spectraData_2])
            if wavelength is None:
                wavelength = np.concatenate([w1, w2])
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

    return spectra, wavelength, len(spectra_list)


def prepare_spectral_tokens(spectra, wavelength, max_seq_len, wl_min, wl_max):
    """
    Convert spectra to token representation: token_i = [intensity_i, wavelength_i].

    Uses saved wl_min/wl_max from training for consistent wavelength normalization.
    If the spectrum is longer than max_seq_len, adjacent points are averaged (binned).

    Returns tokens: (n_samples, seq_len, 2) float32
    """
    n_samples, n_wl = spectra.shape
    wl = wavelength.copy().astype(np.float64)
    spec = spectra.copy()

    if max_seq_len and n_wl > max_seq_len:
        bin_size = n_wl // max_seq_len
        actual_len = max_seq_len * bin_size
        spec = spec[:, :actual_len].reshape(n_samples, max_seq_len, bin_size).mean(axis=2)
        wl = wl[:actual_len].reshape(max_seq_len, bin_size).mean(axis=1)

    wl_norm = ((wl - wl_min) / (wl_max - wl_min + 1e-10)).astype(np.float32)

    tokens = np.zeros((n_samples, len(wl), 2), dtype=np.float32)
    tokens[:, :, 0] = spec
    tokens[:, :, 1] = wl_norm[np.newaxis, :]

    return tokens


def load_model():
    """
    Load trained transformer model from the experiment folder.

    Supports both the full checkpoint format (dict with 'config' key, produced
    by the current training script) and the legacy format (raw state_dict only).
    In the legacy case, reads config.json from the experiment folder as fallback.

    Returns (model, checkpoint) where model is in eval mode on CPU.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: No trained model found: {MODEL_PATH}")
        print("Please run train_nn_autotransformer.py first.")
        sys.exit(1)

    print(f"  Loading model: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    # Handle legacy format: checkpoint is raw state_dict or missing 'config' key
    if not isinstance(checkpoint, dict) or 'config' not in checkpoint:
        config_json_path = os.path.join(os.path.dirname(MODEL_PATH), 'config.json')
        if not os.path.exists(config_json_path):
            print(f"\nERROR: Checkpoint missing 'config' key and no config.json found:")
            print(f"  {config_json_path}")
            print("Please retrain the model (run train_nn_autotransformer.py).")
            sys.exit(1)
        with open(config_json_path) as f:
            json_cfg = json.load(f)
        cfg = {
            'd_model': json_cfg['d_model'],
            'n_heads': json_cfg['n_heads'],
            'n_layers': json_cfg['n_encoder_layers'],
            'dim_ff': json_cfg['dim_feedforward'],
            'n_elements': json_cfg['n_elements'],
            'branch_hidden': json_cfg['branch_hidden_size'],
            'n_bins': json_cfg.get('n_bins', N_BINS),
            'max_seq_len': json_cfg['seq_len'],
            'dropout': json_cfg.get('dropout', 0.1),
        }
        # Wrap a bare state_dict into the expected checkpoint structure
        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
            state_dict = checkpoint
        else:
            state_dict = checkpoint['model_state_dict']
        checkpoint = {
            'model_state_dict': state_dict,
            'config': cfg,
            'elements': json_cfg['elements'],
            'wl_min': json_cfg['wl_min'],
            'wl_max': json_cfg['wl_max'],
        }
    else:
        cfg = checkpoint['config']

    model = SpectralTransformerNN(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        dim_ff=cfg['dim_ff'],
        n_elements=cfg['n_elements'],
        branch_hidden=cfg['branch_hidden'],
        n_bins=cfg.get('n_bins', N_BINS),
        max_seq_len=cfg['max_seq_len'],
        dropout=cfg.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


# ============================================================================
# Prediction
# ============================================================================

def predict_elements(tokens, model):
    """
    Predict element concentrations from spectral tokens.

    Returns:
        concentrations: (n_samples, n_elements) predicted concentration fractions [0, 1]
        pred_bins:      (n_samples, n_elements) predicted bin indices [0, N_BINS-1]
    """
    with torch.no_grad():
        logits = model(torch.FloatTensor(tokens)).numpy()

    pred_bins = np.argmax(logits, axis=2)
    concentrations = bin_to_concentration(pred_bins)

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
    print("Element Identification - Transformer Model Prediction")
    print("=" * 70)

    # 1. Load model
    print(f"\nLoading trained model...")
    model, checkpoint = load_model()
    cfg = checkpoint['config']
    element_names = checkpoint['elements']
    print(f"  Model loaded successfully")
    print(f"  Elements: {len(element_names)}")
    print(f"  Transformer: d_model={cfg['d_model']}, heads={cfg['n_heads']}, "
          f"layers={cfg['n_layers']}")

    # 2. Load input spectra from JSON
    print(f"\nLoading input spectra from: {input_file}")
    input_spectra, wavelength, n_valid_runs = load_input_spectra(input_file)
    print(f"  Valid runs: {n_valid_runs}")
    print(f"  Spectra shape: {input_spectra.shape}")
    print(f"  Wavelength range: [{wavelength.min():.2f}, {wavelength.max():.2f}] nm")

    # 3. Prepare spectral tokens
    print(f"\nPreparing spectral tokens...")
    tokens = prepare_spectral_tokens(
        input_spectra, wavelength,
        max_seq_len=cfg['max_seq_len'],
        wl_min=checkpoint['wl_min'],
        wl_max=checkpoint['wl_max'],
    )
    print(f"  Token shape: {tokens.shape}")

    # 4. Predict
    print(f"\nComputing predictions...")
    concentrations, pred_bins = predict_elements(tokens, model)
    print(f"  Predictions: {concentrations.shape}")

    # 5. Output summary
    print(f"\n{'='*70}")
    print(f"RESULTS - {concentrations.shape[0]} spectra, threshold={threshold*100:.1f}%")
    print(f"{'='*70}")

    for i in range(concentrations.shape[0]):
        conc = concentrations[i]
        detected = [
            (element_names[j], conc[j])
            for j in range(len(conc)) if conc[j] > threshold
        ]
        detected.sort(key=lambda x: x[1], reverse=True)
        top_idx = np.argsort(conc)[::-1][:5]
        top_str = ', '.join([
            f"{element_names[j]}:{conc[j]*100:.1f}%" for j in top_idx
        ])
        print(f"\nSpectrum #{i+1}:")
        print(f"  Detected ({len(detected)}): "
              f"{[(e, f'{c*100:.1f}%') for e, c in detected] if detected else 'None'}")
        print(f"  Top 5: {top_str}")

    os.makedirs(VALIDATION_PATH, exist_ok=True)
    if output_file:
        output_path = os.path.join(VALIDATION_PATH, output_file)
        save_to_csv(concentrations, element_names, output_path)

    print_spectrum_prediction(concentrations[0], element_names, 0, threshold)

    print("\n" + "=" * 70)
    print("Prediction Complete!")
    print("=" * 70)

    return concentrations, element_names


if __name__ == "__main__":
    concentrations, elements = main()
