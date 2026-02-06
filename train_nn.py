"""
Neural Network Training for Element Identification

This script:
1. Loads synthetic spectra from synthetic_data.h5
2. Loads weight spectra from element_weights.h5
3. Performs matrix multiplication: spectra @ weights.T
4. Trains a fully connected neural network with one hidden layer
5. Predicts element presence (0-1) for all 77 elements

Uses PyTorch if available, otherwise falls back to optimized NumPy.
"""

import numpy as np
import h5py
import os
import pickle

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    USE_TORCH = True
    print("Using PyTorch backend")
except ImportError:
    USE_TORCH = False
    print("PyTorch not available, using NumPy backend")

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DATA_PATH = os.path.join(SCRIPT_DIR, 'synthetic_data.h5')
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, 'element_weights', 'element_weights.h5')

# Neural network parameters
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.01
EPOCHS = 1000
BATCH_SIZE = 8
TEST_SPLIT = 0.2
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
if USE_TORCH:
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_data():
    """Load spectra and ground truth from HDF5 files."""
    print("Loading data from HDF5 files...")
    
    with h5py.File(SYNTHETIC_DATA_PATH, 'r') as f:
        synthetic_spectra = f['measurements/Measurement_1/libs/data'][:]
        concentrations = f['measurements/Measurement_1/libs/metadata/concentrations'][:]
        synthetic_elements = [e.decode('utf-8') for e in f['measurements/Measurement_1/libs/metadata/elements'][:]]
    
    print(f"  Synthetic spectra shape: {synthetic_spectra.shape}")
    print(f"  Concentrations shape: {concentrations.shape}")
    print(f"  Synthetic elements ({len(synthetic_elements)}): {synthetic_elements}")
    
    with h5py.File(WEIGHTS_PATH, 'r') as f:
        weight_spectra = f['measurements/Measurement_1/libs/data'][:]
        weight_elements = [e.decode('utf-8') for e in f['measurements/Measurement_1/libs/metadata/elements'][:]]
    
    print(f"  Weight spectra shape: {weight_spectra.shape}")
    print(f"  Weight elements ({len(weight_elements)}): {weight_elements}")
    
    return synthetic_spectra, weight_spectra, concentrations, synthetic_elements, weight_elements


def prepare_features(synthetic_spectra, weight_spectra):
    """Apply matrix multiplication: spectra @ weights.T"""
    print("\nComputing feature matrix via matrix multiplication...")
    print(f"  synthetic_spectra: {synthetic_spectra.shape}")
    print(f"  weight_spectra.T: {weight_spectra.T.shape}")
    
    features = synthetic_spectra @ weight_spectra.T
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Result features: {features.shape}")
    return features


def prepare_targets_77_elements(concentrations, synthetic_elements, weight_elements, threshold=0.0):
    """
    Create target matrix for all 77 weight elements.
    Output values: 0 = element not present, 1 = element present
    """
    n_samples = concentrations.shape[0]
    n_weight_elements = len(weight_elements)
    
    element_to_idx = {elem: idx for idx, elem in enumerate(weight_elements)}
    targets = np.zeros((n_samples, n_weight_elements), dtype=np.float32)
    
    print("\nMapping synthetic elements to weight element indices:")
    for i, elem in enumerate(synthetic_elements):
        if elem in element_to_idx:
            idx = element_to_idx[elem]
            targets[:, idx] = (concentrations[:, i] > threshold).astype(np.float32)
            print(f"  {elem} -> index {idx}")
        else:
            print(f"  {elem} -> NOT FOUND!")
    
    print(f"\nPrepared targets for all {n_weight_elements} elements:")
    print(f"  Shape: {targets.shape}")
    print(f"  Elements present per sample: {targets.sum(axis=1).mean():.2f} average")
    
    return targets


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and test sets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    
    return X[indices[n_test:]], X[indices[:n_test]], y[indices[n_test:]], y[indices[:n_test]]


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
    class ElementIdentificationNN(nn.Module):
        """Fully connected NN with one hidden layer for element identification."""
        
        def __init__(self, n_inputs, n_hidden, n_outputs, dropout=0.2):
            super(ElementIdentificationNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(n_inputs, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_hidden, n_outputs),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x)

    def train_torch(model, train_loader, val_loader, epochs):
        """Train using PyTorch."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf')
        best_state = None
        
        print("\nTraining with PyTorch...")
        print(f"Device: {device}")
        print("-" * 70)
        
        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += ((outputs > 0.5).float() == targets).sum().item()
                train_total += targets.numel()
            
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
                    val_correct += ((outputs > 0.5).float() == targets).sum().item()
                    val_total += targets.numel()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if best_state:
            model.load_state_dict(best_state)
        
        print("-" * 70)
        return history


# ============================================================================
# NumPy Implementation
# ============================================================================

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def bce_loss(y_pred, y_true, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class NumpyNN:
    """Fully connected NN implemented in NumPy."""
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_outputs) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_outputs))
        self.cache = {}
    
    def forward(self, X):
        self.cache['Z1'] = X @ self.W1 + self.b1
        self.cache['A1'] = relu(self.cache['Z1'])
        self.cache['Z2'] = self.cache['A1'] @ self.W2 + self.b2
        self.cache['A2'] = sigmoid(self.cache['Z2'])
        return self.cache['A2']
    
    def backward(self, X, y, lr, clip=1.0):
        n = X.shape[0]
        
        dZ2 = self.cache['A2'] - y
        dW2 = np.clip(self.cache['A1'].T @ dZ2 / n, -clip, clip)
        db2 = np.clip(np.mean(dZ2, axis=0, keepdims=True), -clip, clip)
        
        dZ1 = (dZ2 @ self.W2.T) * relu_derivative(self.cache['Z1'])
        dW1 = np.clip(X.T @ dZ1 / n, -clip, clip)
        db1 = np.clip(np.mean(dZ1, axis=0, keepdims=True), -clip, clip)
        
        self.W2 -= lr * np.nan_to_num(dW2)
        self.b2 -= lr * np.nan_to_num(db2)
        self.W1 -= lr * np.nan_to_num(dW1)
        self.b1 -= lr * np.nan_to_num(db1)


def train_numpy(model, X_train, y_train, X_val, y_val, epochs, lr):
    """Train using NumPy."""
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\nTraining with NumPy...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Forward and backward
        train_pred = model.forward(X_train)
        model.backward(X_train, y_train, lr)
        
        train_loss = bce_loss(train_pred, y_train)
        train_acc = ((train_pred > 0.5) == y_train).mean()
        
        val_pred = model.forward(X_val)
        val_loss = bce_loss(val_pred, y_val)
        val_acc = ((val_pred > 0.5) == y_val).mean()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    print("-" * 70)
    return history


# ============================================================================
# Evaluation and Prediction
# ============================================================================

def evaluate_model(predict_fn, X_test, y_test, element_names):
    """Evaluate model and print results."""
    y_pred = predict_fn(X_test)
    y_pred_binary = (y_pred > 0.5).astype(float)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    overall_acc = (y_pred_binary == y_test).mean()
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    # Active elements (present in at least one test sample)
    active = np.where(y_test.sum(axis=0) > 0)[0]
    
    print(f"\nActive elements in test set ({len(active)}):")
    print("-" * 70)
    print(f"{'Element':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for idx in active:
        elem = element_names[idx]
        pred = y_pred_binary[:, idx]
        true = y_test[:, idx]
        
        acc = (pred == true).mean()
        tp = ((pred == 1) & (true == 1)).sum()
        pp = pred.sum()
        ap = true.sum()
        
        prec = tp / pp if pp > 0 else 0
        rec = tp / ap if ap > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"{elem:<10} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    print("-" * 70)
    
    # Sample predictions
    print("\nSample Predictions (first 3 test samples):")
    print("-" * 70)
    
    for i in range(min(3, len(y_test))):
        true_elems = [element_names[j] for j in range(len(element_names)) if y_test[i, j] > 0]
        pred_elems = [element_names[j] for j in range(len(element_names)) if y_pred_binary[i, j] > 0]
        
        top_idx = np.argsort(y_pred[i])[::-1][:10]
        top_probs = [(element_names[j], y_pred[i, j]) for j in top_idx]
        
        print(f"\nSample {i+1}:")
        print(f"  True: {true_elems}")
        print(f"  Pred: {pred_elems}")
        print(f"  Top 10 probs: {[(e, f'{p:.3f}') for e, p in top_probs]}")
    
    return y_pred


def print_all_77_predictions(predict_fn, X_sample, y_true, element_names, sample_idx=0):
    """
    Print predictions for all 77 elements for a single sample with ground truth.
    
    Parameters
    ----------
    predict_fn : callable
        Prediction function
    X_sample : np.ndarray
        Single sample features
    y_true : np.ndarray
        Ground truth for this sample (77 values)
    element_names : list
        List of element names
    sample_idx : int
        Index of the sample being displayed
    """
    y_pred = predict_fn(X_sample.reshape(1, -1))[0]
    
    print("\n" + "=" * 80)
    print(f"ALL 77 ELEMENTS - SAMPLE #{sample_idx + 1}")
    print("=" * 80)
    
    # Show ground truth elements
    true_elements = [element_names[i] for i in range(len(element_names)) if y_true[i] > 0.5]
    print(f"\nGround Truth Elements: {true_elements}")
    print(f"Number of elements present: {len(true_elements)}")
    
    # Sort by probability (descending)
    sorted_idx = np.argsort(y_pred)[::-1]
    
    print(f"\n{'Element':<8} {'Prediction':<12} {'Ground Truth':<14} {'Status':<12} {'Match'}")
    print("-" * 65)
    
    correct = 0
    total = len(element_names)
    
    for idx in sorted_idx:
        elem = element_names[idx]
        prob = y_pred[idx]
        truth = y_true[idx]
        
        pred_status = "PRESENT" if prob > 0.5 else "absent"
        truth_status = "PRESENT" if truth > 0.5 else "absent"
        
        # Check if prediction matches ground truth
        pred_binary = 1 if prob > 0.5 else 0
        truth_binary = 1 if truth > 0.5 else 0
        match = "✓" if pred_binary == truth_binary else "✗"
        
        if pred_binary == truth_binary:
            correct += 1
        
        print(f"{elem:<8} {prob:<12.4f} {truth:<14.1f} {pred_status:<12} {match}")
    
    print("-" * 65)
    print(f"\nSample Accuracy: {correct}/{total} = {correct/total:.2%}")
    
    # Summary of mismatches
    mismatches = []
    for idx in range(len(element_names)):
        pred_binary = 1 if y_pred[idx] > 0.5 else 0
        truth_binary = 1 if y_true[idx] > 0.5 else 0
        if pred_binary != truth_binary:
            mismatches.append((element_names[idx], y_pred[idx], y_true[idx]))
    
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        for elem, pred, truth in mismatches:
            print(f"  {elem}: predicted {pred:.4f}, truth {truth:.1f}")
    else:
        print("\nNo mismatches - Perfect prediction!")
    
    return {elem: float(y_pred[i]) for i, elem in enumerate(element_names)}


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Neural Network Training for Element Identification")
    print(f"Backend: {'PyTorch' if USE_TORCH else 'NumPy'}")
    print("=" * 70)
    
    # Load data
    synthetic_spectra, weight_spectra, concentrations, synthetic_elements, weight_elements = load_data()
    
    # Prepare features and targets
    features = prepare_features(synthetic_spectra, weight_spectra)
    targets = prepare_targets_77_elements(concentrations, synthetic_elements, weight_elements)
    
    # Split data
    print(f"\nSplitting data (test size: {TEST_SPLIT})...")
    X_train, X_test, y_train, y_test = train_test_split(features, targets, TEST_SPLIT, RANDOM_SEED)
    print(f"  Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Normalize
    print("\nNormalizing features...")
    X_train_norm, X_test_norm, feat_mean, feat_std = normalize_features(X_train, X_test)
    
    n_inputs = features.shape[1]   # 77
    n_outputs = len(weight_elements)  # 77
    
    print(f"\nNetwork architecture:")
    print(f"  Input: {n_inputs} -> Hidden: {HIDDEN_LAYER_SIZE} -> Output: {n_outputs}")
    print(f"  Output: 77 values from 0 to 1 (0=absent, 1=present)")
    
    if USE_TORCH:
        # PyTorch training
        model = ElementIdentificationNN(n_inputs, HIDDEN_LAYER_SIZE, n_outputs).to(device)
        print(f"\nModel:\n{model}")
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_norm), torch.FloatTensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_norm), torch.FloatTensor(y_test)),
            batch_size=BATCH_SIZE
        )
        
        history = train_torch(model, train_loader, val_loader, EPOCHS)
        
        def predict_fn(X):
            model.eval()
            with torch.no_grad():
                return model(torch.FloatTensor(X).to(device)).cpu().numpy()
        
        # Save PyTorch model
        model_path = os.path.join(SCRIPT_DIR, 'element_identification_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_mean': feat_mean,
            'feature_std': feat_std,
            'weight_elements': weight_elements,
            'config': {'n_inputs': n_inputs, 'n_hidden': HIDDEN_LAYER_SIZE, 'n_outputs': n_outputs}
        }, model_path)
        
    else:
        # NumPy training
        model = NumpyNN(n_inputs, HIDDEN_LAYER_SIZE, n_outputs)
        history = train_numpy(model, X_train_norm, y_train, X_test_norm, y_test, EPOCHS, LEARNING_RATE)
        
        def predict_fn(X):
            return model.forward(X)
        
        # Save NumPy model
        model_path = os.path.join(SCRIPT_DIR, 'element_identification_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'W1': model.W1, 'b1': model.b1, 'W2': model.W2, 'b2': model.b2,
                'feature_mean': feat_mean, 'feature_std': feat_std,
                'weight_elements': weight_elements,
                'config': {'n_inputs': n_inputs, 'n_hidden': HIDDEN_LAYER_SIZE, 'n_outputs': n_outputs}
            }, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate
    predictions = evaluate_model(predict_fn, X_test_norm, y_test, weight_elements)
    
    # Print all 77 element predictions for first test sample with ground truth
    print_all_77_predictions(predict_fn, X_test_norm[0], y_test[0], weight_elements, sample_idx=0)
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final Train Acc: {history['train_acc'][-1]:.4f}")
    print(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
    print(f"  Output: 77 elements, each with probability 0-1")
    print("=" * 70)
