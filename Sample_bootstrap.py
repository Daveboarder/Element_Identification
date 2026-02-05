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

# Add Source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source'))
from SpectraGenerator import create_spectra

# ============================================================================
# Configuration
# ============================================================================

# Database path
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'Source', 'LIBS_data.db')

# Number of synthetic samples to generate
N_SAMPLES = 100

#read sample wavelengths from h5 file
file_path = '/home/LIBS/prochazka/data/Running_projects/25_0069_3D_chemical_imaging/Measurements/mandible 266nm/mandible_266_v1.h5'
with h5py.File(file_path, 'r') as file:
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]

# Plasma temperature range (Kelvin)
TE_MIN = 8000
TE_MAX = 20000

# Electron number density range (cm^-3)
NE_MIN = 1e16
NE_MAX = 1e19

# Elements that are invalid or should be excluded from database
EXCLUDED_ELEMENTS = {'', 'n', 'r'}

# ============================================================================
# Element Concentration Ranges
# ============================================================================
# Define concentration ranges for each element you want to include.
# Format: 'Element': (min_concentration, max_concentration)
# Concentrations are given as fractions (0 = 0%, 1 = 100%)
# 
# IMPORTANT: Only elements specified here will have non-zero concentrations.
#            All other elements will have concentration = 0.
# ============================================================================

ELEMENT_CONCENTRATION_RANGES = {
    # Major elements (higher concentrations)
    'Fe': (0.0, 0.5),
    'Si': (0.0, 0.3),
    'Al': (0.0, 0.3),
    'Ca': (0.0, 0.2),
    'Mg': (0.0, 0.2),
    'Na': (0.0, 0.1),
    'K': (0.0, 0.1),
    # Trace elements (lower concentrations)
    'Cu': (0.0, 0.01),
    'Zn': (0.0, 0.01),
    'Pb': (0.0, 0.005),
    'Cd': (0.0, 0.001),
}

# Optical path length (cm)
OPTICAL_PATH_LENGTH = 1.4e-04

# Number density (cm^-3)
NUMBER_DENSITY = 1e-4


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
        - Element concentrations (only for specified elements)
        - Te (plasma temperature)
        - Ne (electron number density)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Create concentration data only for specified elements
    data = {}
    for elem, (c_min, c_max) in concentration_ranges.items():
        data[elem] = np.random.uniform(c_min, c_max, n_samples)
    
    # Add plasma parameters
    data['Te'] = np.random.uniform(te_range[0], te_range[1], n_samples)
    # Use log-uniform distribution for Ne (spans several orders of magnitude)
    log_ne_min, log_ne_max = np.log10(ne_range[0]), np.log10(ne_range[1])
    data['Ne'] = 10 ** np.random.uniform(log_ne_min, log_ne_max, n_samples)
    
    return pd.DataFrame(data)


def generate_synthetic_spectra(sample_table: pd.DataFrame,
                                wavelength: np.ndarray,
                                n_density: float = NUMBER_DENSITY,
                                optical_path: float = OPTICAL_PATH_LENGTH,
                                verbose: bool = True) -> np.ndarray:
    """
    Generate synthetic spectra for all samples in the table.
    
    Parameters
    ----------
    sample_table : pd.DataFrame
        Table with element concentrations and plasma parameters.
        Element columns are automatically detected (all columns except Te, Ne).
    wavelength : np.ndarray
        Wavelength array (nm)
    n_density : float
        Number density (cm^-3)
    optical_path : float
        Optical path length (cm)
    verbose : bool
        Print progress information
        
    Returns
    -------
    np.ndarray
        2D array of shape (n_samples, n_wavelengths) containing spectra
    """
    n_samples = len(sample_table)
    n_wavelengths = len(wavelength)
    spectra = np.zeros((n_samples, n_wavelengths))
    
    # Get element columns (all columns except plasma parameters)
    plasma_params = {'Te', 'Ne'}
    elements = [col for col in sample_table.columns if col not in plasma_params]
    
    if verbose:
        print(f"   Elements to process: {elements}")
    
    for i, row in sample_table.iterrows():
        if verbose and i % 10 == 0:
            print(f"Generating spectrum {i+1}/{n_samples}...")
        
        Te = row['Te']
        Ne = row['Ne']
        
        # Sum contributions from all elements with non-zero concentration
        spectrum = np.zeros(n_wavelengths)
        for elem in elements:
            concentration = row[elem]
            if concentration > 0:
                try:
                    elem_spectrum = create_spectra(
                        element=elem,
                        wavelength=wavelength,
                        Te=Te,
                        Ne=Ne,
                        N=n_density,
                        C=concentration,
                        l=optical_path
                    )
                    spectrum += elem_spectrum
                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not generate spectrum for {elem}: {e}")
        
        spectra[i] = spectrum
    
    if verbose:
        print(f"Generated {n_samples} synthetic spectra.")
    
    return spectra


def save_results(sample_table: pd.DataFrame,
                 spectra: np.ndarray,
                 wavelength: np.ndarray,
                 output_dir: str = None):
    """
    Save the sample table and spectra to files.
    
    Parameters
    ----------
    sample_table : pd.DataFrame
        Table with element concentrations and plasma parameters
    spectra : np.ndarray
        Array of synthetic spectra
    wavelength : np.ndarray
        Wavelength array
    output_dir : str, optional
        Output directory (defaults to script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    # Save sample table
    table_path = os.path.join(output_dir, 'sample_table.csv')
    sample_table.to_csv(table_path, index_label='sample_id')
    print(f"Sample table saved to: {table_path}")
    
    # Save spectra
    spectra_path = os.path.join(output_dir, 'synthetic_spectra.npy')
    np.save(spectra_path, spectra)
    print(f"Spectra saved to: {spectra_path}")
    
    # Save wavelength array
    wavelength_path = os.path.join(output_dir, 'wavelength.npy')
    np.save(wavelength_path, wavelength)
    print(f"Wavelength array saved to: {wavelength_path}")
    
    # Save combined data as HDF5 for convenience
    try:
        import h5py
        h5_path = os.path.join(output_dir, 'synthetic_data.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('spectra', data=spectra)
            f.create_dataset('wavelength', data=wavelength)
            # Save sample table as separate datasets
            for col in sample_table.columns:
                f.create_dataset(f'samples/{col}', data=sample_table[col].values)
        print(f"Combined HDF5 file saved to: {h5_path}")
    except ImportError:
        print("h5py not available, skipping HDF5 output")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic LIBS Spectra Generator")
    print("=" * 70)
    
    # 1. Validate specified elements exist in database
    print("\n1. Validating element configuration...")
    elements = validate_elements(ELEMENT_CONCENTRATION_RANGES, DATABASE_PATH)
    print(f"   Using {len(elements)} elements: {', '.join(elements)}")
    print(f"   (All other elements will have concentration = 0)")
    
    # 2. Show concentration ranges
    print("\n2. Concentration ranges:")
    for elem, (c_min, c_max) in ELEMENT_CONCENTRATION_RANGES.items():
        print(f"   {elem}: {c_min*100:.2f}% - {c_max*100:.2f}%")
    
    # 3. Generate sample table
    print("\n3. Generating sample composition table...")
    sample_table = generate_sample_table(
        concentration_ranges=ELEMENT_CONCENTRATION_RANGES,
        n_samples=N_SAMPLES,
        te_range=(TE_MIN, TE_MAX),
        ne_range=(NE_MIN, NE_MAX),
        random_seed=42  # For reproducibility
    )
    print(f"   Generated {len(sample_table)} samples")
    print(f"   Te range: {sample_table['Te'].min():.0f} - {sample_table['Te'].max():.0f} K")
    print(f"   Ne range: {sample_table['Ne'].min():.2e} - {sample_table['Ne'].max():.2e} cm^-3")
    
    # 4. Create wavelength array
    print("\n4. Creating wavelength array...")
    
    # 5. Generate synthetic spectra
    print("\n5. Generating synthetic spectra...")
    spectra = generate_synthetic_spectra(
        sample_table=sample_table,
        wavelength=wavelength,
        verbose=True
    )
    
    # 6. Save results
    print("\n6. Saving results...")
    save_results(sample_table, spectra, wavelength)
    
    print("\n" + "=" * 70)
    print("Done! Generated synthetic LIBS spectra for {} samples".format(N_SAMPLES))
    print("=" * 70)
    
    # Display summary statistics
    print("\nSample Table Summary:")
    print(sample_table.describe())
