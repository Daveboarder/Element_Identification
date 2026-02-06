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
from SpectraGenerator import create_spectra  # type: ignore[import-not-found]

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
# Sample Type Definitions
# ============================================================================
# Define multiple sample types, each with:
# - 'sample_id': Unique identifier for the sample type
# - 'sample_name': Human-readable name for the sample type
# - 'n_samples': Number of samples to generate for this type
# - 'concentration_ranges': Dict of element concentration ranges {element: (min, max)}
#
# Format for concentration_ranges: 'Element': (min_concentration, max_concentration)
# Concentrations are given as fractions (0 = 0%, 1 = 100%)
# 
# IMPORTANT: Only elements specified will have non-zero concentrations.
#            All other elements will have concentration = 0.
# ============================================================================

SAMPLE_TYPES = [
    {
        'sample_id': 'STEEL_001',
        'sample_name': 'Steel Alloy',
        'n_samples': 50,
        'concentration_ranges': {
            # Major elements for steel
            'Fe': (0.85, 0.98),
            'C': (0.001, 0.02),
            'Mn': (0.0, 0.02),
            'Si': (0.0, 0.01),
            'Cr': (0.0, 0.02),
            'Ni': (0.0, 0.01),
            # Trace elements
            'Cu': (0.0, 0.005),
            'Mo': (0.0, 0.005),
        }
    },
    {
        'sample_id': 'ROCK_001',
        'sample_name': 'Geological Rock Sample',
        'n_samples': 50,
        'concentration_ranges': {
            # Major elements for rocks/minerals
            'Si': (0.2, 0.4),
            'Al': (0.05, 0.15),
            'Fe': (0.02, 0.1),
            'Ca': (0.01, 0.15),
            'Mg': (0.01, 0.1),
            'Na': (0.01, 0.05),
            'K': (0.01, 0.05),
            # Trace elements
            'Ti': (0.0, 0.01),
            'Mn': (0.0, 0.005),
        }
    },
    {
        'sample_id': 'BIO_001',
        'sample_name': 'Biological Tissue',
        'n_samples': 50,
        'concentration_ranges': {
            # Major elements for biological samples
            'Ca': (0.1, 0.4),
            'P': (0.05, 0.2),
            'Mg': (0.001, 0.01),
            'Na': (0.001, 0.01),
            'K': (0.001, 0.01),
            # Trace elements
            'Fe': (0.0, 0.001),
            'Zn': (0.0, 0.0005),
            'Cu': (0.0, 0.0001),
        }
    },
]

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
    
    # Get element columns (all columns except plasma parameters and metadata)
    non_element_cols = {'Te', 'Ne', 'sample_type_id', 'sample_type_name', 'unique_id'}
    elements = [col for col in sample_table.columns if col not in non_element_cols]
    
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
    # Following the structure from weight_generator.py
    try:
        import h5py
        h5_path = os.path.join(output_dir, 'synthetic_data.h5')
        with h5py.File(h5_path, 'w') as f:
            # Create group structure
            f.create_group('measurements')
            f.create_group('measurements/Measurement_1')
            f.create_group('measurements/Measurement_1/libs')
            f.create_group('measurements/Measurement_1/libs/metadata')
            f.create_group('measurements/Measurement_1/libs/metadata/samples')
            f.create_group('measurements/Measurement_1/global_metadata')
            
            # Main data
            f.create_dataset('measurements/Measurement_1/libs/data', data=spectra)
            f.create_dataset('measurements/Measurement_1/libs/calibration', data=wavelength)
            
            # Identify element columns vs metadata columns
            non_element_cols = {'Te', 'Ne', 'sample_type_id', 'sample_type_name', 'unique_id'}
            element_cols = [col for col in sample_table.columns if col not in non_element_cols]
            
            # Save element names
            elements_encoded = np.array(element_cols, dtype='S10')
            f.create_dataset('measurements/Measurement_1/libs/metadata/elements', data=elements_encoded)
            
            # Save sample metadata (per-sample arrays)
            for col in sample_table.columns:
                col_data = sample_table[col].values
                # Handle string columns by encoding to bytes
                if col_data.dtype == object or col_data.dtype.kind == 'U':
                    col_data = np.array(col_data, dtype='S64')
                f.create_dataset(f'measurements/Measurement_1/libs/metadata/samples/{col}', data=col_data)
            
            # Save element concentrations as a 2D array (n_samples x n_elements)
            concentration_matrix = sample_table[element_cols].values
            f.create_dataset('measurements/Measurement_1/libs/metadata/concentrations', data=concentration_matrix)
            
            # Save plasma parameters
            f.create_dataset('measurements/Measurement_1/libs/metadata/Te', data=sample_table['Te'].values)
            f.create_dataset('measurements/Measurement_1/libs/metadata/Ne', data=sample_table['Ne'].values)
            
            # Global metadata (placeholder values)
            f.create_dataset('measurements/Measurement_1/libs/metadata/x', data=np.zeros(len(sample_table)))
            f.create_dataset('measurements/Measurement_1/libs/metadata/y', data=np.zeros(len(sample_table)))
            f.create_dataset('measurements/Measurement_1/libs/metadata/z', data=np.zeros(len(sample_table)))
            f.create_dataset('measurements/Measurement_1/libs/metadata/X_pos', data=np.zeros(len(sample_table)))
            f.create_dataset('measurements/Measurement_1/libs/metadata/Y_pos', data=np.zeros(len(sample_table)))
            f.create_dataset('measurements/Measurement_1/global_metadata/Width Spacing', data=0)
            f.create_dataset('measurements/Measurement_1/global_metadata/Height Spacing', data=0)
            f.create_dataset('measurements/Measurement_1/global_metadata/optical_path_length', data=OPTICAL_PATH_LENGTH)
            f.create_dataset('measurements/Measurement_1/global_metadata/number_density', data=NUMBER_DENSITY)
            
        print(f"Combined HDF5 file saved to: {h5_path}")
    except ImportError:
        print("h5py not available, skipping HDF5 output")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic LIBS Spectra Generator - Multi-Sample Type Mode")
    print("=" * 70)
    
    # Lists to collect all samples and spectra
    all_sample_tables = []
    all_spectra = []
    
    # Process each sample type
    for i, sample_type in enumerate(SAMPLE_TYPES):
        sample_id = sample_type['sample_id']
        sample_name = sample_type['sample_name']
        n_samples = sample_type['n_samples']
        concentration_ranges = sample_type['concentration_ranges']
        
        print(f"\n{'='*70}")
        print(f"Processing Sample Type {i+1}/{len(SAMPLE_TYPES)}: {sample_name} ({sample_id})")
        print(f"{'='*70}")
        
        # 1. Validate elements
        print(f"\n1. Validating elements for {sample_name}...")
        try:
            elements = validate_elements(concentration_ranges, DATABASE_PATH)
            print(f"   Using {len(elements)} elements: {', '.join(elements)}")
        except ValueError as e:
            print(f"   WARNING: Skipping {sample_name} - {e}")
            continue
        
        # 2. Show concentration ranges
        print(f"\n2. Concentration ranges for {sample_name}:")
        for elem, (c_min, c_max) in concentration_ranges.items():
            print(f"   {elem}: {c_min*100:.3f}% - {c_max*100:.3f}%")
        
        # 3. Generate sample table
        print(f"\n3. Generating {n_samples} samples for {sample_name}...")
        sample_table = generate_sample_table(
            concentration_ranges=concentration_ranges,
            n_samples=n_samples,
            sample_id=sample_id,
            sample_name=sample_name,
            te_range=(TE_MIN, TE_MAX),
            ne_range=(NE_MIN, NE_MAX),
            random_seed=42 + i  # Different seed for each sample type
        )
        print(f"   Generated {len(sample_table)} samples")
        print(f"   Te range: {sample_table['Te'].min():.0f} - {sample_table['Te'].max():.0f} K")
        print(f"   Ne range: {sample_table['Ne'].min():.2e} - {sample_table['Ne'].max():.2e} cm^-3")
        
        # 4. Generate synthetic spectra
        print(f"\n4. Generating synthetic spectra for {sample_name}...")
        spectra = generate_synthetic_spectra(
            sample_table=sample_table,
            wavelength=wavelength,
            verbose=True
        )
        
        # Collect results
        all_sample_tables.append(sample_table)
        all_spectra.append(spectra)
    
    # Combine all sample tables and spectra
    print(f"\n{'='*70}")
    print("Combining all sample types...")
    print(f"{'='*70}")
    
    if all_sample_tables:
        # Combine sample tables (fill missing columns with 0 for elements)
        combined_sample_table = pd.concat(all_sample_tables, ignore_index=True)
        combined_sample_table = combined_sample_table.fillna(0)
        
        # Combine spectra
        combined_spectra = np.vstack(all_spectra)
        
        print(f"\nTotal samples generated: {len(combined_sample_table)}")
        print(f"Sample types: {combined_sample_table['sample_type_id'].nunique()}")
        for sample_id in combined_sample_table['sample_type_id'].unique():
            count = (combined_sample_table['sample_type_id'] == sample_id).sum()
            name = combined_sample_table[combined_sample_table['sample_type_id'] == sample_id]['sample_type_name'].iloc[0]
            print(f"   - {sample_id} ({name}): {count} samples")
        
        # Save results
        print("\nSaving combined results...")
        save_results(combined_sample_table, combined_spectra, wavelength)
        
        print("\n" + "=" * 70)
        print(f"Done! Generated synthetic LIBS spectra for {len(combined_sample_table)} total samples")
        print(f"Across {len(SAMPLE_TYPES)} sample types")
        print("=" * 70)
        
        # Display summary statistics
        print("\nCombined Sample Table Summary:")
        print(combined_sample_table.describe())
    else:
        print("\nNo samples were generated. Check the sample type configurations.")
