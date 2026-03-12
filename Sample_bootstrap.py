"""
Sample Bootstrap Generator for Synthetic LIBS Spectra

This script generates a set of synthetic LIBS (Laser-Induced Breakdown Spectroscopy) 
spectra based on a sample composition table. Each sample has:
- Random concentrations for each element within specified ranges
- Randomly selected plasma temperature (Te)
- Randomly selected electron number density (Ne)

The output is a table where:
- Each column represents one element
- Each row represents one artificial sample
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import sys
import h5py
import torch
from torch.utils.data import Dataset  # type: ignore[import-not-found]

# Add Source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source'))
from SpectraGenerator import create_spectra  # type: ignore[import-not-found]
from read_sample_types import load_sample_types_from_excel
from readData import get_spectra


# ============================================================================
# Configuration
# ============================================================================

# Database path
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'Source', 'LIBS_data_vacuum.db')

# Number of synthetic samples to generate
N_SAMPLES = 100

#read sample wavelengths from json file
file_path = '/mnt/data/projects/Running_projects/26_0128_Element_Identification/Methods/Element_Identification-1/Data/VASKUT K8.json'
data, wavelength = get_spectra(file_path, run_id=1)

# Plasma temperature range (Kelvin)
TE_MIN = 8000
TE_MAX = 20000

# Electron number density range (cm^-3)
NE_MIN = 1e16
NE_MAX = 1e19

# Elements that are invalid or should be excluded from database
EXCLUDED_ELEMENTS = {'', 'n', 'r'}

# ============================================================================
# Sample Type Definitions (loaded from external config file)
# ============================================================================
# To use different sample types, edit sample_types_config.py or point to
# a different config file. This keeps the main script reusable across
# different sample type sets without losing previous configurations.
# ============================================================================

SAMPLE_TYPES = load_sample_types_from_excel()

# Optical path length (cm)
OPTICAL_PATH_LENGTH = 1.4e-04

# Number density (cm^-3)
NUMBER_DENSITY = 1e-4

def unit_norm(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to have unit length.
    """
    vector = vector - np.min(vector)
    vmax = np.max(vector)
    if vmax == 0:
        return vector
    return vector / vmax

def get_elements_from_database(db_path: str) -> list:
    """
    Retrieve all unique element names from the LIBS database.
    
    Parameters
    ----------
    db_path : str
        Path to the SQLite database
        
    Returns
    -------
    list
        List of valid element names (excluding ionized species marked with -II)
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT Elem_name FROM QuantParam ORDER BY Elem_name')
        elements = [row[0] for row in cursor.fetchall()]
    
    # Filter out invalid elements and ionized species (they are handled internally)
    valid_elements = [e for e in elements if e not in EXCLUDED_ELEMENTS and '-II' not in e]
    
    return valid_elements


def validate_elements(concentration_ranges: dict, db_path: str) -> list:
    """
    Validate that all elements in concentration_ranges exist in the database.
    
    Parameters
    ----------
    concentration_ranges : dict
        Dictionary with element concentration ranges: {element: (min, max)}
    db_path : str
        Path to the SQLite database
        
    Returns
    -------
    list
        List of valid element names that exist in the database
        
    Raises
    ------
    ValueError
        If any element in concentration_ranges is not found in the database
    """
    # Get all valid elements from database
    all_db_elements = get_elements_from_database(db_path)
    
    # Check which specified elements exist in database
    valid_elements = []
    missing_elements = []
    
    for elem in concentration_ranges.keys():
        if elem in all_db_elements:
            valid_elements.append(elem)
        else:
            missing_elements.append(elem)
    
    if missing_elements:
        raise ValueError(
            f"The following elements are not in the database: {missing_elements}\n"
            f"Available elements: {all_db_elements}"
        )
    
    return valid_elements


def generate_sample_table(concentration_ranges: dict,
                          n_samples: int,
                          sample_id: str = None,
                          sample_name: str = None,
                          te_range: tuple = (TE_MIN, TE_MAX),
                          ne_range: tuple = (NE_MIN, NE_MAX),
                          random_seed: int = None) -> pd.DataFrame:
    """
    Generate a table of synthetic samples with random element concentrations
    and plasma parameters.
    
    Only elements specified in concentration_ranges will have non-zero values.
    All other elements are assumed to have concentration = 0.
    
    Parameters
    ----------
    concentration_ranges : dict
        Dictionary with concentration ranges for each element: {element: (min, max)}
        Only elements in this dictionary will be included in the table.
    n_samples : int
        Number of samples to generate
    sample_id : str, optional
        Unique identifier for this sample type
    sample_name : str, optional
        Human-readable name for this sample type
    te_range : tuple
        (min, max) temperature range in Kelvin
    ne_range : tuple
        (min, max) electron number density range in cm^-3
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame where each row is a sample and columns are:
        - sample_type_id: Unique identifier for the sample type
        - sample_type_name: Human-readable name for the sample type
        - unique_id: Unique identifier for each individual sample
        - Element concentrations (only for specified elements)
        - Te (plasma temperature)
        - Ne (electron number density)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create sample identification columns
    data = {}
    if sample_id is not None:
        data['sample_type_id'] = [sample_id] * n_samples
        # Create unique IDs for each sample: sample_id_001, sample_id_002, etc.
        data['unique_id'] = [f"{sample_id}_{i+1:04d}" for i in range(n_samples)]
    if sample_name is not None:
        data['sample_type_name'] = [sample_name] * n_samples
    
    # Create concentration data only for specified elements
    element_names = list(concentration_ranges.keys())
    concentration_matrix = np.column_stack([
        np.random.uniform(c_min, c_max, n_samples)
        for c_min, c_max in concentration_ranges.values()
    ])

    # Normalize each sample (row) so concentrations sum to 1.0
    row_sums = concentration_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    concentration_matrix = concentration_matrix / row_sums

    for idx, elem in enumerate(element_names):
        data[elem] = concentration_matrix[:, idx]
    
    # Add plasma parameters
    data['Te'] = np.random.uniform(te_range[0], te_range[1], n_samples)
    # Use log-uniform distribution for Ne (spans several orders of magnitude)
    log_ne_min, log_ne_max = np.log10(ne_range[0]), np.log10(ne_range[1])
    data['Ne'] = 10 ** np.random.uniform(log_ne_min, log_ne_max, n_samples)
    
    return pd.DataFrame(data)


import multiprocessing as _mp

# ---------------------------------------------------------------------------
# Worker state – set once per worker process via Pool initializer
# ---------------------------------------------------------------------------
_w_wavelength: np.ndarray | None = None
_w_db_path: str | None = None
_w_n_density: float = 0.0
_w_optical_path: float = 0.0


def _init_worker(wavelength: np.ndarray, db_path: str,
                 n_density: float, optical_path: float) -> None:
    """Called once in each worker process to set shared state."""
    global _w_wavelength, _w_db_path, _w_n_density, _w_optical_path
    _w_wavelength = wavelength
    _w_db_path = db_path
    _w_n_density = n_density
    _w_optical_path = optical_path

    # Reset SpectraGenerator caches so forked connections are not reused.
    import SpectraGenerator as _sg
    _sg._db_connection = None
    _sg._quant_cache.clear()
    _sg._eion_cache.clear()

    import LIBSmethods as _lm
    _lm._partf_conn = None
    _lm._partf_data_cache.clear()


def _generate_one(args: tuple) -> tuple[int, np.ndarray]:
    """Generate a single sample spectrum (runs inside a worker process)."""
    idx, elements, concentrations, Te_i, Ne_i = args
    spectrum = np.zeros(len(_w_wavelength))
    for j, elem in enumerate(elements):
        if concentrations[j] > 0:
            try:
                spectrum += create_spectra(
                    element=elem,
                    wavelength=_w_wavelength,
                    Te=Te_i,
                    Ne=Ne_i,
                    N=_w_n_density,
                    C=concentrations[j],
                    l=_w_optical_path,
                    db_path=_w_db_path,
                )
            except Exception:
                pass
    return idx, unit_norm(spectrum)


def generate_synthetic_spectra(sample_table: pd.DataFrame,
                                wavelength: np.ndarray,
                                db_path: str = None,
                                n_density: float = NUMBER_DENSITY,
                                optical_path: float = OPTICAL_PATH_LENGTH,
                                verbose: bool = True,
                                n_workers: int = None) -> np.ndarray:
    """
    Generate synthetic spectra for all samples in the table.

    Parameters
    ----------
    sample_table : pd.DataFrame
        Table with element concentrations and plasma parameters.
    wavelength : np.ndarray
        Wavelength array (nm)
    db_path : str, optional
        Path to LIBS SQLite database (forwarded to create_spectra).
    n_density : float
        Number density (cm^-3)
    optical_path : float
        Optical path length (cm)
    verbose : bool
        Print progress information
    n_workers : int, optional
        Number of parallel workers (defaults to cpu_count).
        Set to 1 to disable multiprocessing.

    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, n_wavelengths) containing spectra
    """
    if db_path is None:
        db_path = DATABASE_PATH

    n_samples = len(sample_table)
    n_wavelengths = len(wavelength)
    spectra = np.zeros((n_samples, n_wavelengths))

    non_element_cols = {'Te', 'Ne', 'sample_type_id', 'sample_type_name', 'unique_id'}
    elements = [col for col in sample_table.columns if col not in non_element_cols]

    if verbose:
        print(f"   Elements to process: {elements}")

    Te_arr = sample_table['Te'].values
    Ne_arr = sample_table['Ne'].values
    conc_matrix = sample_table[elements].values

    if n_workers is None:
        n_workers = min(_mp.cpu_count(), n_samples)

    # Build lightweight per-sample argument tuples
    tasks = [
        (i, elements, conc_matrix[i], Te_arr[i], Ne_arr[i])
        for i in range(n_samples)
    ]

    if n_workers > 1 and n_samples > 1:
        if verbose:
            print(f"   Parallelising across {n_workers} workers...")
        with _mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(wavelength, db_path, n_density, optical_path),
        ) as pool:
            for idx, spectrum in pool.imap_unordered(_generate_one, tasks, chunksize=4):
                spectra[idx] = spectrum
                if verbose and (idx + 1) % 200 == 0:
                    print(f"   Completed {idx + 1}/{n_samples}")
    else:
        # Sequential fallback
        _init_worker(wavelength, db_path, n_density, optical_path)
        for task in tasks:
            idx, spectrum = _generate_one(task)
            spectra[idx] = spectrum
            if verbose and idx % 100 == 0:
                print(f"Generating spectrum {idx + 1}/{n_samples}...")

    if verbose:
        print(f"Generated {n_samples} synthetic spectra.")

    return spectra


class SyntheticLIBSDataset(Dataset):
    """
    PyTorch Dataset wrapper for synthetic LIBS sample generation.
    """

    def __init__(self,
                 sample_types: list,
                 wavelength: np.ndarray,
                 db_path: str = DATABASE_PATH,
                 te_range: tuple = (TE_MIN, TE_MAX),
                 ne_range: tuple = (NE_MIN, NE_MAX),
                 verbose: bool = True):
        self.sample_types = sample_types
        self.wavelength = wavelength
        self.db_path = db_path
        self.te_range = te_range
        self.ne_range = ne_range
        self.verbose = verbose
        self.sample_table, self.spectra = self._generate_all()

    def _generate_all(self) -> tuple[pd.DataFrame, np.ndarray]:
        # Phase 1 -- build all sample tables (fast, sequential)
        all_sample_tables = []
        for i, sample_type in enumerate(self.sample_types):
            sample_id = sample_type['sample_id']
            sample_name = sample_type['sample_name']
            n_samples = sample_type['n_samples']
            concentration_ranges = sample_type['concentration_ranges']

            try:
                validate_elements(concentration_ranges, self.db_path)
            except ValueError as e:
                if self.verbose:
                    print(f"   WARNING: Skipping {sample_name} - {e}")
                continue

            sample_table = generate_sample_table(
                concentration_ranges=concentration_ranges,
                n_samples=n_samples,
                sample_id=sample_id,
                sample_name=sample_name,
                te_range=self.te_range,
                ne_range=self.ne_range,
                random_seed=42 + i,
            )
            all_sample_tables.append(sample_table)

        if not all_sample_tables:
            return pd.DataFrame(), np.empty((0, len(self.wavelength)))

        combined_sample_table = pd.concat(all_sample_tables, ignore_index=True).fillna(0)

        if self.verbose:
            print(f"\nTotal samples to generate spectra for: {len(combined_sample_table)}")

        # Phase 2 -- generate all spectra in one parallelised batch
        combined_spectra = generate_synthetic_spectra(
            sample_table=combined_sample_table,
            wavelength=self.wavelength,
            db_path=self.db_path,
            verbose=self.verbose,
        )

        return combined_sample_table, combined_spectra

    def __len__(self) -> int:
        return len(self.sample_table)

    def __getitem__(self, idx: int) -> dict:
        row = self.sample_table.iloc[idx].to_dict()
        row['spectrum'] = self.spectra[idx]
        return row


# Disabled legacy save helper (kept as comments intentionally).
# def save_results(sample_table: pd.DataFrame,
#                  spectra: np.ndarray,
#                  wavelength: np.ndarray,
#                  output_dir: str = None):
#     ...

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic LIBS Spectra Generator - Multi-Sample Type Mode")
    print("=" * 70)

    dataset = SyntheticLIBSDataset(
        sample_types=SAMPLE_TYPES,
        wavelength=wavelength,
        db_path=DATABASE_PATH,
        te_range=(TE_MIN, TE_MAX),
        ne_range=(NE_MIN, NE_MAX),
        verbose=True
    )

    combined_sample_table = dataset.sample_table
    combined_spectra = dataset.spectra

    print(f"\n{'='*70}")
    print("Combining all sample types...")
    print(f"{'='*70}")

    if len(dataset) > 0:
        print(f"\nTotal samples generated: {len(combined_sample_table)}")
        print(f"Sample types: {combined_sample_table['sample_type_id'].nunique()}")
        for sample_id in combined_sample_table['sample_type_id'].unique():
            count = (combined_sample_table['sample_type_id'] == sample_id).sum()
            name = combined_sample_table[combined_sample_table['sample_type_id'] == sample_id]['sample_type_name'].iloc[0]
            print(f"   - {sample_id} ({name}): {count} samples")

        # Save results
        #print("\nSaving combined results...")
        #save_results(combined_sample_table, combined_spectra, wavelength)

        print("\n" + "=" * 70)
        print(f"Done! Generated synthetic LIBS spectra for {len(combined_sample_table)} total samples")
        print(f"Across {len(SAMPLE_TYPES)} sample types")
        print("=" * 70)

        # Display summary statistics
        print("\nCombined Sample Table Summary:")
        print(combined_sample_table.describe())
    else:
        print("\nNo samples were generated. Check the sample type configurations.")
