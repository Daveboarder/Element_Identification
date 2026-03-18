"""
Neural Network Training for Element Identification (Classification Approach)

Based on train_nn.py, but reformulated as classification:
- Each element's concentration [0%, 100%] is discretized into 1000 bins
- Bin mapping: bin_index = round(concentration_fraction * 999)
      0%   -> bin 0   (position 1)
      75%  -> bin 749  (position 750)
      100% -> bin 999  (position 1000)
- Each element has its own classification branch
- Network predicts the correct bin for each element independently

Architecture:
    spectra @ weight_spectra.T -> (n, 7700) @ W_proj -> (n, 77) -> SharedHidden
    -> 77 separate branches, each: Linear -> ReLU -> Linear -> 1000 logits
    Loss: CrossEntropy per element, averaged

Prerequisite:
    Run weight_generator.py first to generate element_weights/multi_weights_vacuum.h5

Uses PyTorch if available, otherwise falls back to NumPy.
"""

import numpy as np
import h5py
import os
import sys
import pickle
import Sample_bootstrap as sample_bootstrap

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    import torch.optim as optim  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import-not-found]
    USE_TORCH = True
    print("Using PyTorch backend")
except ImportError:
    USE_TORCH = False
    print("PyTorch not available, using NumPy backend")

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTI_WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'element_weights', 'multi_weights_vacuum.h5')

N_BINS = 1000
HIDDEN_LAYER_SIZE = 256
BRANCH_HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 8
TEST_SPLIT = 0.2
RANDOM_SEED = 17

np.random.seed(RANDOM_SEED)
if USE_TORCH:
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

def load_multi_weights():
    """Load multi-weight spectra from element_weights/multi_weights_vacuum.h5."""
    if not os.path.exists(MULTI_WEIGHTS_PATH):
        print(f"\nERROR: Multi-weight file not found: {MULTI_WEIGHTS_PATH}")
        print("Please run weight_generator.py first to generate the weight spectra.")
        sys.exit(1)

    print(f"Loading multi-weight spectra from: {MULTI_WEIGHTS_PATH}")
    with h5py.File(MULTI_WEIGHTS_PATH, 'r') as f:
        weight_matrix = f['weight_matrix'][:]
        unique_elements = [e.decode('utf-8') for e in f['unique_elements'][:]]
        element_labels = [e.decode('utf-8') for e in f['element_labels'][:]]
        te_values = f['te_values'][:]
        ne_values = f['ne_values'][:]

    n_elements = len(unique_elements)
    n_combos = len(te_values) * len(ne_values)

    print(f"  Weight matrix: {weight_matrix.shape}")
    print(f"  Elements: {n_elements}")
    print(f"  TE/NE combinations: {n_combos}")
    print(f"  Total weight spectra: {n_combos} x {n_elements} = {weight_matrix.shape[0]}")

    return weight_matrix, unique_elements


def load_synthetic_data_from_dataloader(batch_size=64):
    """Generate synthetic spectra/targets using SyntheticLIBSDataset and DataLoader."""
    print("\nGenerating synthetic data via SyntheticLIBSDataset...")
    synthetic_dataset = sample_bootstrap.SyntheticLIBSDataset(
        sample_types=sample_bootstrap.SAMPLE_TYPES,
        wavelength=sample_bootstrap.wavelength,
        db_path=sample_bootstrap.DATABASE_PATH,
        te_range=(sample_bootstrap.TE_MIN, sample_bootstrap.TE_MAX),
        ne_range=(sample_bootstrap.NE_MIN, sample_bootstrap.NE_MAX),
        verbose=False
    )

    if len(synthetic_dataset) == 0:
        print("\nERROR: Synthetic dataset is empty. Check SAMPLE_TYPES configuration.")
        sys.exit(1)

    non_element_cols = {'Te', 'Ne', 'sample_type_id', 'sample_type_name', 'unique_id'}
    synth_elements = [
        col for col in synthetic_dataset.sample_table.columns
        if col not in non_element_cols
    ]

    concentrations_np = synthetic_dataset.sample_table[synth_elements].to_numpy(dtype=np.float32)
    spectra_np = synthetic_dataset.spectra.astype(np.float32)

    if USE_TORCH:
        data_loader = DataLoader(
            TensorDataset(torch.FloatTensor(spectra_np), torch.FloatTensor(concentrations_np)),
            batch_size=batch_size, shuffle=False, num_workers=0
        )
        spectra_batches, concentration_batches = [], []
        for spectra_batch, concentrations_batch in data_loader:
            spectra_batches.append(spectra_batch.numpy())
            concentration_batches.append(concentrations_batch.numpy())
        spectra = np.vstack(spectra_batches)
        concentrations = np.vstack(concentration_batches)
    else:
        spectra = spectra_np
        concentrations = concentrations_np

    print(f"  Spectra: {spectra.shape}")
    print(f"  Concentrations: {concentrations.shape}")
    print(f"  Elements ({len(synth_elements)}): {synth_elements}")

    return spectra, concentrations, synth_elements


def prepare_features(synthetic_spectra, weight_matrix):
    """Apply matrix multiplication: spectra @ weights.T"""
    print(f"\nComputing feature matrix via matrix multiplication...")
    print(f"  synthetic_spectra: {synthetic_spectra.shape}")
    print(f"  weight_matrix.T: {weight_matrix.T.shape}")

    features = synthetic_spectra @ weight_matrix.T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Result features: {features.shape}")
    return features


def prepare_targets_binned(concentrations, synth_elements, weight_elements):
    """
    Convert concentrations to bin indices for classification.
    Returns (n_samples, n_weight_elements) with int64 bin indices [0, N_BINS-1].
    Elements not present in the sample get bin 0 (concentration 0%).
    """
    n_samples = concentrations.shape[0]
    n_weight_elements = len(weight_elements)
    element_to_idx = {elem: idx for idx, elem in enumerate(weight_elements)}

    targets = np.zeros((n_samples, n_weight_elements), dtype=np.int64)

    print(f"\nMapping concentrations to {N_BINS} bins:")
    mapped_count = 0
    for i, elem in enumerate(synth_elements):
        if elem in element_to_idx:
            idx = element_to_idx[elem]
            conc = np.nan_to_num(
                concentrations[:, i], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            conc = np.clip(conc, 0.0, 1.0)
            targets[:, idx] = concentration_to_bin(conc)
            mapped_count += 1
        else:
            print(f"  {elem} -> NOT FOUND in weight elements!")

    print(f"  Mapped {mapped_count}/{len(synth_elements)} elements")
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


def normalize_features(X_train, X_test):
    """Normalize features using training set statistics."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1
    X_train_norm = np.nan_to_num((X_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_norm = np.nan_to_num((X_test - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train_norm, X_test_norm, mean, std


# ============================================================================
# PyTorch Implementation
# ============================================================================

if USE_TORCH:
    class ElementBranchNN(nn.Module):
        """
        NN with shared trunk and per-element classification branches.

        Shared: projection (n_features -> n_elements) -> BatchNorm -> ReLU -> Dropout
        Per-element branch: Linear(n_hidden, branch_hidden) -> ReLU -> Linear(branch_hidden, n_bins)

        Each branch outputs N_BINS logits; softmax is applied by CrossEntropyLoss.
        """

        def __init__(self, n_features, n_elements, n_hidden, branch_hidden,
                     n_bins=N_BINS, dropout=0.2):
            super(ElementBranchNN, self).__init__()
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


    def train_torch(model, train_loader, val_loader, epochs):
        """Train using PyTorch with per-element CrossEntropyLoss."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.5, patience=30
        )

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_mae': [], 'val_mae': [],
        }
        best_val_loss = float('inf')
        best_state = None

        print(f"\nTraining with PyTorch on {device}...")
        print("-" * 80)

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            train_abs_err = 0.0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)  # (batch, n_elements, n_bins)

                loss = criterion(
                    outputs.reshape(-1, model.n_bins),
                    targets.reshape(-1),
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred_bins = outputs.argmax(dim=2)
                train_correct += (pred_bins == targets).sum().item()
                train_total += targets.numel()
                train_abs_err += torch.abs(
                    pred_bins.float() - targets.float()
                ).sum().item()

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

            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)

            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:4d}/{epochs}: "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                    f"MAE: {train_mae:.4f}/{val_mae:.4f}"
                )

        if best_state:
            model.load_state_dict(best_state)
        print("-" * 80)
        return history


# ============================================================================
# NumPy Implementation
# ============================================================================

def np_softmax(x):
    """Numerically stable softmax along last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def cross_entropy_loss(probs, target_indices):
    """Cross-entropy loss from softmax probabilities and integer targets."""
    n = len(target_indices)
    log_probs = -np.log(probs[np.arange(n), target_indices] + 1e-10)
    return np.mean(log_probs)


class NumpyBranchNN:
    """
    NN with shared trunk and per-element classification branches (NumPy).

    Architecture mirrors ElementBranchNN:
        projection -> shared hidden -> per-element (hidden -> ReLU -> output)
    """

    def __init__(self, n_features, n_elements, n_hidden, branch_hidden, n_bins):
        self.n_elements = n_elements
        self.n_bins = n_bins

        self.W0 = np.random.randn(n_features, n_elements) * np.sqrt(2.0 / n_features)
        self.W1 = np.random.randn(n_elements, n_hidden) * np.sqrt(2.0 / n_elements)
        self.b1 = np.zeros((1, n_hidden))

        self.branch_W1 = [
            np.random.randn(n_hidden, branch_hidden) * np.sqrt(2.0 / n_hidden)
            for _ in range(n_elements)
        ]
        self.branch_b1 = [np.zeros((1, branch_hidden)) for _ in range(n_elements)]
        self.branch_W2 = [
            np.random.randn(branch_hidden, n_bins) * np.sqrt(2.0 / branch_hidden)
            for _ in range(n_elements)
        ]
        self.branch_b2 = [np.zeros((1, n_bins)) for _ in range(n_elements)]

        self.cache = {}

    def forward(self, X):
        self.cache['X'] = X
        self.cache['P'] = X @ self.W0
        self.cache['Z1'] = self.cache['P'] @ self.W1 + self.b1
        self.cache['A1'] = relu(self.cache['Z1'])

        self.cache['bZ1'] = []
        self.cache['bA1'] = []
        self.cache['bLogits'] = []
        self.cache['bProbs'] = []

        for e in range(self.n_elements):
            bz1 = self.cache['A1'] @ self.branch_W1[e] + self.branch_b1[e]
            ba1 = relu(bz1)
            logits = ba1 @ self.branch_W2[e] + self.branch_b2[e]
            probs = np_softmax(logits)
            self.cache['bZ1'].append(bz1)
            self.cache['bA1'].append(ba1)
            self.cache['bLogits'].append(logits)
            self.cache['bProbs'].append(probs)

        return self.cache['bProbs']

    def backward(self, X, y, lr, clip=1.0):
        """y: (batch, n_elements) int64 bin indices."""
        n = X.shape[0]
        dA1_total = np.zeros_like(self.cache['A1'])

        for e in range(self.n_elements):
            one_hot = np.zeros((n, self.n_bins), dtype=np.float64)
            one_hot[np.arange(n), y[:, e]] = 1.0
            dLogits = (self.cache['bProbs'][e] - one_hot) / n

            dBW2 = np.clip(self.cache['bA1'][e].T @ dLogits, -clip, clip)
            dBb2 = np.clip(np.sum(dLogits, axis=0, keepdims=True), -clip, clip)

            dBA1 = dLogits @ self.branch_W2[e].T
            dBZ1 = dBA1 * relu_derivative(self.cache['bZ1'][e])
            dBW1 = np.clip(self.cache['A1'].T @ dBZ1, -clip, clip)
            dBb1 = np.clip(np.sum(dBZ1, axis=0, keepdims=True), -clip, clip)

            dA1_total += dBZ1 @ self.branch_W1[e].T

            self.branch_W2[e] -= lr * np.nan_to_num(dBW2)
            self.branch_b2[e] -= lr * np.nan_to_num(dBb2)
            self.branch_W1[e] -= lr * np.nan_to_num(dBW1)
            self.branch_b1[e] -= lr * np.nan_to_num(dBb1)

        dZ1 = dA1_total * relu_derivative(self.cache['Z1'])
        dW1 = np.clip(self.cache['P'].T @ dZ1, -clip, clip)
        db1 = np.clip(np.sum(dZ1, axis=0, keepdims=True), -clip, clip)

        dP = dZ1 @ self.W1.T
        dW0 = np.clip(self.cache['X'].T @ dP, -clip, clip)

        self.W1 -= lr * np.nan_to_num(dW1)
        self.b1 -= lr * np.nan_to_num(db1)
        self.W0 -= lr * np.nan_to_num(dW0)


def train_numpy(model, X_train, y_train, X_val, y_val, epochs, lr):
    """Train using NumPy."""
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_mae': [], 'val_mae': [],
    }
    print("\nTraining with NumPy...")
    print("-" * 80)

    for epoch in range(epochs):
        train_probs = model.forward(X_train)
        model.backward(X_train, y_train, lr)

        train_loss = np.mean([
            cross_entropy_loss(train_probs[e], y_train[:, e])
            for e in range(model.n_elements)
        ])
        train_preds = np.stack([np.argmax(p, axis=1) for p in train_probs], axis=1)
        train_acc = (train_preds == y_train).mean()
        train_mae = np.abs(train_preds - y_train).mean() / (N_BINS - 1)

        val_probs = model.forward(X_val)
        val_loss = np.mean([
            cross_entropy_loss(val_probs[e], y_val[:, e])
            for e in range(model.n_elements)
        ])
        val_preds = np.stack([np.argmax(p, axis=1) for p in val_probs], axis=1)
        val_acc = (val_preds == y_val).mean()
        val_mae = np.abs(val_preds - y_val).mean() / (N_BINS - 1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs}: "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                f"MAE: {train_mae:.4f}/{val_mae:.4f}"
            )

    print("-" * 80)
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
    print("Neural Network Training - Classification Approach")
    print(f"Backend: {'PyTorch' if USE_TORCH else 'NumPy'}")
    print(f"Bins per element: {N_BINS}")
    print("=" * 80)

    # 1. Load multi-weight spectra
    weight_matrix, unique_elements = load_multi_weights()

    # 2. Generate synthetic training data
    synth_spectra, concentrations, synth_elements = load_synthetic_data_from_dataloader(
        batch_size=BATCH_SIZE
    )

    # 3. Check wavelength compatibility
    if synth_spectra.shape[1] != weight_matrix.shape[1]:
        print(f"\nERROR: Wavelength mismatch!")
        print(f"  Synthetic spectra: {synth_spectra.shape[1]} wavelengths")
        print(f"  Weight spectra: {weight_matrix.shape[1]} wavelengths")
        sys.exit(1)

    # 4. Prepare features (same as train_nn.py)
    features = prepare_features(synth_spectra, weight_matrix)

    # 5. Prepare binned targets
    targets = prepare_targets_binned(concentrations, synth_elements, unique_elements)

    # 6. Split data
    print(f"\nSplitting data (test size: {TEST_SPLIT})...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, TEST_SPLIT, RANDOM_SEED
    )
    print(f"  Training: {len(X_train)}, Test: {len(X_test)}")

    # 7. Normalize features
    print("\nNormalizing features...")
    X_train_norm, X_test_norm, feat_mean, feat_std = normalize_features(X_train, X_test)

    n_features = features.shape[1]
    n_elements = len(unique_elements)

    print(f"\nNetwork architecture:")
    print(f"  Features: {n_features} (from weight matrix multiplication)")
    print(f"  Projection: {n_features} -> {n_elements} (trainable)")
    print(f"  Shared hidden: {n_elements} -> {HIDDEN_LAYER_SIZE}")
    print(f"  Per-element branch ({n_elements}x): "
          f"{HIDDEN_LAYER_SIZE} -> {BRANCH_HIDDEN_SIZE} -> {N_BINS} logits")

    # 8. Train
    if USE_TORCH:
        model = ElementBranchNN(
            n_features, n_elements, HIDDEN_LAYER_SIZE, BRANCH_HIDDEN_SIZE, N_BINS
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal model parameters: {n_params:,}")

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_norm), torch.LongTensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_norm), torch.LongTensor(y_test)),
            batch_size=BATCH_SIZE,
        )

        history = train_torch(model, train_loader, val_loader, EPOCHS)

        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                return model(torch.FloatTensor(X).to(device)).cpu().numpy()

        model_path = os.path.join(SCRIPT_DIR, 'element_classification_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_mean': feat_mean,
            'feature_std': feat_std,
            'weight_elements': unique_elements,
            'config': {
                'n_features': n_features,
                'n_elements': n_elements,
                'n_hidden': HIDDEN_LAYER_SIZE,
                'branch_hidden': BRANCH_HIDDEN_SIZE,
                'n_bins': N_BINS,
            }
        }, model_path)
    else:
        model = NumpyBranchNN(
            n_features, n_elements, HIDDEN_LAYER_SIZE, BRANCH_HIDDEN_SIZE, N_BINS
        )
        history = train_numpy(
            model, X_train_norm, y_train, X_test_norm, y_test, EPOCHS, LEARNING_RATE
        )

        def predict_fn(X):
            probs = model.forward(X)
            return np.stack(probs, axis=1)

        model_path = os.path.join(SCRIPT_DIR, 'element_classification_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'W0': model.W0,
                'W1': model.W1, 'b1': model.b1,
                'branch_W1': model.branch_W1, 'branch_b1': model.branch_b1,
                'branch_W2': model.branch_W2, 'branch_b2': model.branch_b2,
                'feature_mean': feat_mean, 'feature_std': feat_std,
                'weight_elements': unique_elements,
                'config': {
                    'n_features': n_features,
                    'n_elements': n_elements,
                    'n_hidden': HIDDEN_LAYER_SIZE,
                    'branch_hidden': BRANCH_HIDDEN_SIZE,
                    'n_bins': N_BINS,
                }
            }, f)

    print(f"\nModel saved to: {model_path}")

    # 9. Evaluate
    predictions = evaluate_model(predict_fn, X_test_norm, y_test, unique_elements)
    print_sample_predictions(predict_fn, X_test_norm, y_test, unique_elements)

    # 10. Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Approach: Classification ({N_BINS} bins per element)")
    print(f"  Weight spectra: {weight_matrix.shape[0]} rows "
          f"({weight_matrix.shape[0] // len(unique_elements)} combos x "
          f"{len(unique_elements)} elements)")
    print(f"  Raw features: {n_features}")
    print(f"  Elements (branches): {n_elements}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss:   {history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc:  {history['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc:    {history['val_acc'][-1]:.4f}")
    print(f"  Final Train MAE:  {history['train_mae'][-1]:.4f}")
    print(f"  Final Val MAE:    {history['val_mae'][-1]:.4f}")
    print("=" * 80)
