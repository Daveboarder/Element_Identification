"""
Neural Network Training for Element Identification (Transformer Approach)

Based on train_nn_classification.py, but replaces weight matrix multiplication
with a Transformer encoder operating on spectral tokens.

Each wavelength point becomes a token: [intensity_i, wavelength_i]
SpectralEmbedding projects each 2D token into d_model dimensions, sinusoidal
positional encoding is added, then a TransformerEncoder processes the sequence.
Mean pooling aggregates to a single vector, which feeds per-element classification
branches (same 1000-bin approach as train_nn_classification.py).

Architecture:
    Input (seq_len, 2) -> SpectralEmbedding -> (seq_len, d_model)
    + SinusoidalPositionalEncoding
    -> TransformerEncoder -> (seq_len, d_model)
    -> MeanPooling + LayerNorm -> (d_model,)
    -> Per-element branches: Linear(d_model, branch_hidden) -> ReLU -> Linear(branch_hidden, 1000)
    Loss: CrossEntropy per element, averaged

Requires PyTorch (no NumPy fallback for transformer).
"""

import json
import numpy as np
import os
import sys
import Sample_bootstrap as sample_bootstrap
import mlflow  # type: ignore[import-not-found]

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.optim as optim  # type: ignore[import-not-found]
from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found]

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_NAME = "element_transformer_test_v5"
DATA_PATH = os.path.join("experiments", EXPERIMENT_NAME)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_BINS = 1000
D_MODEL = 128
N_HEADS = 8
N_ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
MAX_SEQ_LEN = 2048
BRANCH_HIDDEN_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 32
TEST_SPLIT = 0.2
RANDOM_SEED = 17

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Bin Conversion Utilities
# ============================================================================

def concentration_to_bin(concentration):
    """
    Convert concentration fraction [0.0, 1.0] to bin index [0, N_BINS-1].
    0.0 -> 0, 0.75 -> 749, 1.0 -> 999
    """
    return np.clip(
        np.round(np.asarray(concentration, dtype=np.float64) * (N_BINS - 1)).astype(np.int64),
        0, N_BINS - 1
    )


def bin_to_concentration(bin_idx):
    """
    Convert bin index [0, N_BINS-1] to concentration fraction [0.0, 1.0].
    0 -> 0.0, 749 -> 0.7498, 999 -> 1.0
    """
    return np.asarray(bin_idx, dtype=np.float64) / (N_BINS - 1)


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_synthetic_data(batch_size=64):
    """Generate synthetic spectra/targets using SyntheticLIBSDataset."""
    print("\nGenerating synthetic data via SyntheticLIBSDataset...")
    synthetic_dataset = sample_bootstrap.SyntheticLIBSDataset(
        sample_types=sample_bootstrap.SAMPLE_TYPES,
        wavelength=sample_bootstrap.wavelength,
        db_path=sample_bootstrap.DATABASE_PATH,
        te_range=(sample_bootstrap.TE_MIN, sample_bootstrap.TE_MAX),
        ne_range=(sample_bootstrap.NE_MIN, sample_bootstrap.NE_MAX),
        verbose=True
    )

    if len(synthetic_dataset) == 0:
        print("\nERROR: Synthetic dataset is empty. Check SAMPLE_TYPES configuration.")
        sys.exit(1)

    non_element_cols = {'Te', 'Ne', 'sample_type_id', 'sample_type_name', 'unique_id'}
    synth_elements = [
        col for col in synthetic_dataset.sample_table.columns
        if col not in non_element_cols
    ]

    concentrations = synthetic_dataset.sample_table[synth_elements].to_numpy(dtype=np.float32)
    spectra = synthetic_dataset.spectra.astype(np.float32)

    print(f"  Spectra: {spectra.shape}")
    print(f"  Concentrations: {concentrations.shape}")
    print(f"  Elements ({len(synth_elements)}): {synth_elements}")

    return spectra, concentrations, synth_elements


def prepare_spectral_tokens(spectra, wavelength, max_seq_len=MAX_SEQ_LEN):
    """
    Convert spectra to token representation: token_i = [intensity_i, wavelength_i].

    If the spectrum is longer than max_seq_len, adjacent wavelength points are
    averaged (binned) to reduce the sequence length.
    Wavelengths are normalized to [0, 1].

    Returns:
        tokens:     (n_samples, seq_len, 2) float32
        binned_wl:  (seq_len,) the (possibly binned) wavelength array
        wl_min:     original wavelength minimum (for reproducing normalization)
        wl_max:     original wavelength maximum
    """
    n_samples, n_wl = spectra.shape
    wl = wavelength.copy().astype(np.float64)
    spec = spectra.copy()

    print(f"\nPreparing spectral tokens...")
    print(f"  Original spectrum length: {n_wl}")

    if max_seq_len and n_wl > max_seq_len:
        bin_size = n_wl // max_seq_len
        actual_len = max_seq_len * bin_size
        spec = spec[:, :actual_len].reshape(n_samples, max_seq_len, bin_size).mean(axis=2)
        wl = wl[:actual_len].reshape(max_seq_len, bin_size).mean(axis=1)
        print(f"  Binned to {max_seq_len} tokens (bin size: {bin_size}, "
              f"using {actual_len}/{n_wl} points)")

    wl_min, wl_max = float(wl.min()), float(wl.max())
    wl_norm = ((wl - wl_min) / (wl_max - wl_min + 1e-10)).astype(np.float32)

    tokens = np.zeros((n_samples, len(wl), 2), dtype=np.float32)
    tokens[:, :, 0] = spec
    tokens[:, :, 1] = wl_norm[np.newaxis, :]

    print(f"  Token shape: {tokens.shape}")
    print(f"  Wavelength range: [{wl_min:.2f}, {wl_max:.2f}] nm")
    print(f"  Intensity range: [{spec.min():.4f}, {spec.max():.4f}]")

    return tokens, wl, wl_min, wl_max


def prepare_targets_binned(concentrations, synth_elements):
    """
    Convert concentrations directly to bin indices.
    Returns (n_samples, n_elements) with int64 bin indices [0, N_BINS-1].
    """
    conc = np.nan_to_num(concentrations, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    conc = np.clip(conc, 0.0, 1.0)
    targets = concentration_to_bin(conc)

    print(f"\nMapping concentrations to {N_BINS} bins:")
    print(f"  Elements: {len(synth_elements)}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Bin index range: [{targets.min()}, {targets.max()}]")
    conc_range = bin_to_concentration(targets)
    print(f"  Concentration range: [{conc_range.min():.4f}, {conc_range.max():.4f}]")

    return targets


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    idx = np.random.permutation(n_samples)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# ============================================================================
# Model Definition
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
        # x: (batch, seq_len, d_model) -> returns (seq_len, d_model), broadcast over batch
        return self.pe[:x.size(1)]


class SpectralTransformerNN(nn.Module):
    """
    Transformer encoder for spectral element classification.

    Input:  (batch, seq_len, 2) spectral tokens [intensity, wavelength]
    Output: (batch, n_elements, n_bins) logits per element

    Pipeline:
        SpectralEmbedding -> + SinusoidalPE -> Dropout
        -> TransformerEncoder
        -> MeanPool + LayerNorm -> (batch, d_model)
        -> Per-element branches -> (batch, n_elements, n_bins)
    """

    def __init__(self, d_model, n_heads, n_layers, dim_ff, n_elements,
                 branch_hidden, n_bins=N_BINS, max_seq_len=MAX_SEQ_LEN, dropout=0.1):
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
        x = self.embedding(x)                    # (batch, seq_len, d_model)
        x = x + self.pos_encoding(x)             # add positional encoding
        x = self.embed_dropout(x)
        x = self.transformer(x)                  # (batch, seq_len, d_model)
        x = self.pool_norm(x.mean(dim=1))         # mean pool + norm -> (batch, d_model)
        return torch.stack([branch(x) for branch in self.branches], dim=1)


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, val_loader, epochs):
    """Train with per-element CrossEntropyLoss, AdamW optimizer, cosine schedule."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_mae': [], 'val_mae': [],
    }
    best_val_loss = float('inf')
    best_state = None

    # Ensure DATA_PATH directory exists for saving epoch weights
    os.makedirs(DATA_PATH, exist_ok=True)

    # --- MLflow: start a new run and log training parameters ---
    mlflow.set_experiment("element_transformer")
    run = mlflow.start_run(run_name=EXPERIMENT_NAME)
    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "n_bins": N_BINS,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_encoder_layers": N_ENCODER_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "max_seq_len": MAX_SEQ_LEN,
        "branch_hidden_size": BRANCH_HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_split": TEST_SPLIT,
        "random_seed": RANDOM_SEED,
        "experiment_name": EXPERIMENT_NAME,
    })

    print(f"\nTraining on {device}...")
    print(f"MLflow run id: {run.info.run_id}")
    print("-" * 90)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_abs_err = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(
                outputs.reshape(-1, model.n_bins),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pred_bins = outputs.argmax(dim=2)
            train_correct += (pred_bins == targets).sum().item()
            train_total += targets.numel()
            train_abs_err += torch.abs(
                pred_bins.float() - targets.float()
            ).sum().item()

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_abs_err = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)

                val_loss += criterion(
                    outputs.reshape(-1, model.n_bins),
                    targets.reshape(-1),
                ).item()
                pred_bins = outputs.argmax(dim=2)
                val_correct += (pred_bins == targets).sum().item()
                val_total += targets.numel()
                val_abs_err += torch.abs(
                    pred_bins.float() - targets.float()
                ).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        train_mae = (train_abs_err / train_total) / (N_BINS - 1)
        val_mae = (val_abs_err / val_total) / (N_BINS - 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Save best model weights
            best_model_path = os.path.join(DATA_PATH, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        # --- MLflow: log per-epoch metrics ---
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_mae": train_mae,
            "val_mae": val_mae,
        }, step=epoch)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:4d}/{epochs}: "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                f"MAE: {train_mae:.4f}/{val_mae:.4f} | "
                f"LR: {lr:.2e}"
            )

    if best_state:
        model.load_state_dict(best_state)
    mlflow.end_run()
    print("-" * 90)
    return history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(predict_fn, X_test, y_test, element_names):
    """Evaluate model: bin accuracy, MAE, and per-element breakdown."""
    logits = predict_fn(X_test)
    pred_bins = np.argmax(logits, axis=2)

    true_conc = bin_to_concentration(y_test)
    pred_conc = bin_to_concentration(pred_bins)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    exact_acc = (pred_bins == y_test).mean()
    overall_mae = np.abs(pred_conc - true_conc).mean()
    within_5 = (np.abs(pred_bins - y_test) <= 5).mean()
    within_10 = (np.abs(pred_bins - y_test) <= 10).mean()

    print(f"\nOverall exact bin accuracy:  {exact_acc:.4f}")
    print(f"Overall MAE (concentration): {overall_mae:.6f}")
    print(f"Within  5 bins: {within_5:.4f}")
    print(f"Within 10 bins: {within_10:.4f}")

    active = np.where(y_test.max(axis=0) > 0)[0]
    print(f"\nActive elements in test set ({len(active)}):")
    print("-" * 80)
    print(
        f"{'Element':<8} {'ExactAcc':<10} {'MAE(%)':<10} "
        f"{'Within5':<10} {'Within10':<10} {'MeanTrue%':<10} {'MeanPred%':<10}"
    )
    print("-" * 80)

    for idx in active:
        elem = element_names[idx]
        ep = pred_bins[:, idx]
        et = y_test[:, idx]
        acc = (ep == et).mean()
        mae = np.abs(bin_to_concentration(ep) - bin_to_concentration(et)).mean() * 100
        w5 = (np.abs(ep - et) <= 5).mean()
        w10 = (np.abs(ep - et) <= 10).mean()
        mt = bin_to_concentration(et).mean() * 100
        mp = bin_to_concentration(ep).mean() * 100
        print(
            f"{elem:<8} {acc:<10.4f} {mae:<10.2f} "
            f"{w5:<10.4f} {w10:<10.4f} {mt:<10.2f} {mp:<10.2f}"
        )

    print("-" * 80)
    return pred_bins


def print_sample_predictions(predict_fn, X_test, y_test, element_names, n_samples=3):
    """Print detailed per-element predictions for a few test samples."""
    logits = predict_fn(X_test[:n_samples])
    pred_bins = np.argmax(logits, axis=2)

    for i in range(min(n_samples, len(X_test))):
        print(f"\n{'=' * 70}")
        print(f"SAMPLE #{i + 1}")
        print(f"{'=' * 70}")

        active_true = np.where(y_test[i] > 0)[0]
        print(f"\nElements with non-zero concentration ({len(active_true)}):")
        print(
            f"{'Element':<8} {'TrueBin':<10} {'PredBin':<10} "
            f"{'TrueConc%':<12} {'PredConc%':<12} {'Error%':<10}"
        )
        print("-" * 62)

        for idx in active_true:
            tb = int(y_test[i, idx])
            pb = int(pred_bins[i, idx])
            tc = bin_to_concentration(tb) * 100
            pc = bin_to_concentration(pb) * 100
            err = abs(tc - pc)
            print(
                f"{element_names[idx]:<8} {tb:<10d} {pb:<10d} "
                f"{tc:<12.2f} {pc:<12.2f} {err:<10.2f}"
            )


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Neural Network Training - Transformer with Spectral Embedding")
    print(f"Device: {device}")
    print(f"Max sequence length: {MAX_SEQ_LEN}")
    print(f"Transformer: d_model={D_MODEL}, heads={N_HEADS}, "
          f"layers={N_ENCODER_LAYERS}, ff={DIM_FEEDFORWARD}")
    print(f"Bins per element: {N_BINS}")
    print("=" * 80)

    # 1. Get wavelength grid from sample_bootstrap
    wavelength = sample_bootstrap.wavelength
    print(f"\nWavelength grid: {len(wavelength)} points")

    # 2. Generate synthetic training data
    synth_spectra, concentrations, synth_elements = load_synthetic_data(
        batch_size=BATCH_SIZE
    )

    # 3. Check wavelength/spectra compatibility
    if synth_spectra.shape[1] != len(wavelength):
        print(f"\nERROR: Wavelength mismatch!")
        print(f"  Spectra: {synth_spectra.shape[1]}, Wavelength grid: {len(wavelength)}")
        sys.exit(1)

    # 4. Prepare spectral tokens (replaces weight matrix multiplication)
    tokens, binned_wl, wl_min, wl_max = prepare_spectral_tokens(
        synth_spectra, wavelength, MAX_SEQ_LEN
    )
    seq_len = tokens.shape[1]

    # 5. Prepare binned targets
    targets = prepare_targets_binned(concentrations, synth_elements)
    n_elements = len(synth_elements)

    # 6. Split data
    print(f"\nSplitting data (test size: {TEST_SPLIT})...")
    X_train, X_test, y_train, y_test = train_test_split(
        tokens, targets, TEST_SPLIT, RANDOM_SEED
    )
    print(f"  Training: {len(X_train)}, Test: {len(X_test)}")

    # 7. Create model
    model = SpectralTransformerNN(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_ENCODER_LAYERS,
        dim_ff=DIM_FEEDFORWARD,
        n_elements=n_elements,
        branch_hidden=BRANCH_HIDDEN_SIZE,
        n_bins=N_BINS,
        max_seq_len=seq_len,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNetwork architecture:")
    print(f"  Input: ({seq_len}, 2) spectral tokens [intensity, wavelength]")
    print(f"  SpectralEmbedding: 2 -> {D_MODEL}")
    print(f"  + SinusoidalPositionalEncoding (fixed)")
    print(f"  TransformerEncoder: {N_ENCODER_LAYERS} layers, {N_HEADS} heads, "
          f"FF={DIM_FEEDFORWARD}")
    print(f"  MeanPooling + LayerNorm -> ({D_MODEL},)")
    print(f"  Per-element branch ({n_elements}x): "
          f"{D_MODEL} -> {BRANCH_HIDDEN_SIZE} -> {N_BINS} logits")
    print(f"  Total parameters: {n_params:,}")

    # 8. Save config.json before training so it is available even if training fails
    os.makedirs(DATA_PATH, exist_ok=True)
    config_json = {
        'experiment_name': EXPERIMENT_NAME,
        'n_bins': N_BINS,
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'n_encoder_layers': N_ENCODER_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'max_seq_len': MAX_SEQ_LEN,
        'branch_hidden_size': BRANCH_HIDDEN_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'test_split': TEST_SPLIT,
        'random_seed': RANDOM_SEED,
        'n_elements': int(n_elements),
        'seq_len': int(seq_len),
        'wl_min': float(wl_min),
        'wl_max': float(wl_max),
        'elements': list(synth_elements),
    }
    config_json_path = os.path.join(DATA_PATH, 'config.json')
    with open(config_json_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"\nConfig saved to: {config_json_path}")

    # 9. Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=BATCH_SIZE,
    )

    # 10. Train
    history = train_model(model, train_loader, val_loader, EPOCHS)

    # 11. Predict function
    def predict_fn(X):
        model.eval()
        with torch.no_grad():
            return model(torch.FloatTensor(X).to(device)).cpu().numpy()

    # 12. Save final model checkpoints
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'elements': synth_elements,
        'wavelength_binned': binned_wl,
        'wl_min': wl_min,
        'wl_max': wl_max,
        'config': {
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'n_layers': N_ENCODER_LAYERS,
            'dim_ff': DIM_FEEDFORWARD,
            'n_elements': n_elements,
            'branch_hidden': BRANCH_HIDDEN_SIZE,
            'n_bins': N_BINS,
            'max_seq_len': seq_len,
            'dropout': DROPOUT,
        }
    }

    model_path = os.path.join(SCRIPT_DIR, 'element_transformer_model.pt')
    torch.save(checkpoint_data, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save full checkpoint to experiment folder (used by predict_nn.py)
    best_model_path = os.path.join(DATA_PATH, "best_model.pt")
    torch.save(checkpoint_data, best_model_path)
    print(f"Checkpoint saved to: {best_model_path}")

    # 13. Evaluate
    predictions = evaluate_model(predict_fn, X_test, y_test, synth_elements)
    print_sample_predictions(predict_fn, X_test, y_test, synth_elements)

    # 14. Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Approach: Transformer ({N_BINS} bins per element)")
    print(f"  Sequence length: {seq_len} "
          f"(binned from {len(wavelength)} wavelength points)")
    print(f"  Elements (branches): {n_elements}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc:  {history['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc:    {history['val_acc'][-1]:.4f}")
    print(f"  Final Train MAE:  {history['train_mae'][-1]:.4f}")
    print(f"  Final Val MAE:    {history['val_mae'][-1]:.4f}")
    print("=" * 80)
