"""
Weight Generator for Synthetic LIBS Spectra

This script generates synthetic LIBS (Laser-Induced Breakdown Spectroscopy) 
spectra for each element individually with concentration = 1 (100%).
These "weight spectra" can be used as basis functions for spectral analysis.

Supports two modes:
  1. Single TE/NE: One spectrum per element (77 rows)
  2. Multi TE/NE grid: All combinations of TE and NE arrays
     (e.g., 10 TE x 10 NE = 100 combos x 77 elements = 7700 rows)

The output is saved to element_weights/ as HDF5 files.

Configuration:
  - Modify TE, NE values below to adapt to different plasma conditions
  - Modify wavelength source to adapt to different spectrometer setups
  - Modify INCLUDE_ELEMENTS to restrict to specific elements
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

# Read sample wavelength from h5 file
# MODIFY THIS PATH to match your spectrometer / measurement setup
file_path = '/mnt/data/projects/Running_projects/26_0128_Element_Identification/Data/FG_OBSIDIAN.h5'
with h5py.File(file_path, 'r') as file:
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]

# ----------------------------------------------------------------------------
# Plasma Parameters (MODIFY THESE VALUES)
# ----------------------------------------------------------------------------
# Plasma temperature (Kelvin) - array of values for multi-weight generation
TE = np.linspace(7000, 15000, 10)

# Electron number density (cm^-3) - array of values for multi-weight generation
NE = np.linspace(1e16, 1e19, 10)

# ----------------------------------------------------------------------------
# Other Parameters
# ----------------------------------------------------------------------------
# Optical path length (cm)
OPTICAL_PATH_LENGTH = 1.4e-04

# Number density (cm^-3)
NUMBER_DENSITY = 1e-4

# Element concentration (fixed at 1 = 100%)
CONCENTRATION = 1.0

# Elements that are invalid or should be excluded from database
EXCLUDED_ELEMENTS = {'', 'n', 'r'}

# Optional: Specify which elements to include (leave empty to include all)
# Example: INCLUDE_ELEMENTS = ['Fe', 'Si', 'Al', 'Ca', 'Mg']
INCLUDE_ELEMENTS = []  # Empty list = include all available elements


# ============================================================================
# Database Functions
# ============================================================================

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


def filter_elements(all_elements: list, include_list: list) -> list:
    """
    Filter elements based on inclusion list.
    
    Parameters
    ----------
    all_elements : list
        List of all available elements from database
    include_list : list
        List of elements to include (if empty, include all)
        
    Returns
    -------
    list
        Filtered list of elements
        
    Raises
    ------
    ValueError
        If any element in include_list is not found in database
    """
    if not include_list:
        return all_elements
    
    # Check which specified elements exist in database
    missing_elements = [e for e in include_list if e not in all_elements]
    
    if missing_elements:
        raise ValueError(
            f"The following elements are not in the database: {missing_elements}\n"
            f"Available elements: {all_elements}"
        )
    
    return [e for e in include_list if e in all_elements]


# ============================================================================
# Single TE/NE Weight Generation
# ============================================================================

def generate_element_weights(elements: list,
                              wavelength: np.ndarray,
                              Te: float,
                              Ne: float,
                              n_density: float = NUMBER_DENSITY,
                              optical_path: float = OPTICAL_PATH_LENGTH,
                              concentration: float = CONCENTRATION,
                              verbose: bool = True) -> tuple:
    """
    Generate weight spectra for each element with a single TE/NE pair.
    
    Parameters
    ----------
    elements : list
        List of element names to process
    wavelength : np.ndarray
        Wavelength array (nm)
    Te : float
        Plasma temperature (Kelvin)
    Ne : float
        Electron number density (cm^-3)
    n_density : float
        Number density (cm^-3)
    optical_path : float
        Optical path length (cm)
    concentration : float
        Element concentration (default = 1 for weight generation)
    verbose : bool
        Print progress information
        
    Returns
    -------
    tuple
        (weight_spectra, successful_elements, failed_elements)
        - weight_spectra: 2D array of shape (n_elements, n_wavelengths)
        - successful_elements: list of elements that were successfully processed
        - failed_elements: dict of {element: error_message} for failed elements
    """
    successful_elements = []
    failed_elements = {}
    spectra_list = []
    
    for i, elem in enumerate(elements):
        if verbose:
            print(f"  [{i+1}/{len(elements)}] Generating spectrum for {elem}...")
        
        try:
            spectrum = create_spectra(
                element=elem,
                wavelength=wavelength,
                Te=Te,
                Ne=Ne,
                N=n_density,
                C=concentration,
                l=optical_path
            )
            spectra_list.append(spectrum)
            successful_elements.append(elem)
            
        except Exception as e:
            error_msg = str(e)
            failed_elements[elem] = error_msg
            if verbose:
                print(f"      Warning: Could not generate spectrum for {elem}: {error_msg}")
    
    # Stack successful spectra into 2D array
    if spectra_list:
        weight_spectra = np.vstack(spectra_list)
    else:
        weight_spectra = np.array([])
    
    if verbose:
        print(f"\n  Successfully generated {len(successful_elements)} element spectra.")
        if failed_elements:
            print(f"  Failed to generate {len(failed_elements)} element spectra.")
    
    return weight_spectra, successful_elements, failed_elements


# ============================================================================
# Multi TE/NE Weight Generation (7700 spectra)
# ============================================================================

def generate_multi_weights(elements: list,
                            wavelength: np.ndarray,
                            te_values: np.ndarray,
                            ne_values: np.ndarray,
                            n_density: float = NUMBER_DENSITY,
                            optical_path: float = OPTICAL_PATH_LENGTH,
                            concentration: float = CONCENTRATION,
                            verbose: bool = True) -> tuple:
    """
    Generate weight spectra for all TE/NE combinations and all elements.
    
    For each (TE, NE) pair from the grid, generates a spectrum for every
    element. This produces n_combos * n_elements total spectra.
    
    Parameters
    ----------
    elements : list
        List of element names to process
    wavelength : np.ndarray
        Wavelength array (nm)
    te_values : np.ndarray
        Array of plasma temperature values (Kelvin)
    ne_values : np.ndarray
        Array of electron number density values (cm^-3)
    n_density : float
        Number density (cm^-3)
    optical_path : float
        Optical path length (cm)
    concentration : float
        Element concentration (default = 1)
    verbose : bool
        Print progress information

    Returns
    -------
    tuple
        (weight_matrix, element_labels, te_labels, ne_labels)
        - weight_matrix: shape (n_combos * n_elements, n_wavelengths)
        - element_labels: element name for each row
        - te_labels: TE value for each row
        - ne_labels: NE value for each row
    """
    n_combos = len(te_values) * len(ne_values)
    n_elements = len(elements)
    total_spectra = n_combos * n_elements
    
    print(f"\nGenerating {total_spectra} weight spectra:")
    print(f"  {len(te_values)} TE values x {len(ne_values)} NE values = {n_combos} combinations")
    print(f"  {n_combos} combinations x {n_elements} elements = {total_spectra} spectra")
    print(f"  TE range: {te_values[0]:.0f} - {te_values[-1]:.0f} K")
    print(f"  NE range: {ne_values[0]:.2e} - {ne_values[-1]:.2e} cm^-3")
    
    spectra_list = []
    element_labels = []
    te_labels = []
    ne_labels = []
    failed_count = 0
    
    combo_idx = 0
    for te in te_values:
        for ne in ne_values:
            combo_idx += 1
            if verbose:
                print(f"  Combo {combo_idx}/{n_combos}: TE={te:.0f} K, NE={ne:.2e} cm^-3 ...", end="")
            
            combo_ok = 0
            for elem in elements:
                try:
                    spectrum = create_spectra(
                        element=elem,
                        wavelength=wavelength,
                        Te=te,
                        Ne=ne,
                        N=n_density,
                        C=concentration,
                        l=optical_path
                    )
                    spectra_list.append(spectrum)
                    combo_ok += 1
                except Exception:
                    # Use zero spectrum for failed elements
                    spectra_list.append(np.zeros(len(wavelength)))
                    failed_count += 1
                
                element_labels.append(elem)
                te_labels.append(te)
                ne_labels.append(ne)
            
            if verbose:
                print(f" {combo_ok}/{n_elements} elements OK")
    
    weight_matrix = np.vstack(spectra_list)
    
    print(f"\nWeight matrix shape: {weight_matrix.shape}")
    print(f"Failed spectra (replaced with zeros): {failed_count}")
    
    return weight_matrix, element_labels, np.array(te_labels), np.array(ne_labels)


# ============================================================================
# Save Functions
# ============================================================================

def create_element_info_table(elements: list, Te, Ne) -> pd.DataFrame:
    """Create a DataFrame with element information and plasma parameters."""
    data = {
        'element': elements,
        'concentration': [CONCENTRATION] * len(elements),
        'Te': [Te] * len(elements) if np.isscalar(Te) else [str(Te)] * len(elements),
        'Ne': [Ne] * len(elements) if np.isscalar(Ne) else [str(Ne)] * len(elements),
    }
    return pd.DataFrame(data)


def save_single_weights(weight_spectra: np.ndarray,
                         elements: list,
                         wavelength: np.ndarray,
                         Te: float,
                         Ne: float,
                         output_dir: str = None):
    """
    Save single TE/NE weight spectra to element_weights/element_weights.h5
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    output_subdir = os.path.join(output_dir, 'element_weights')
    os.makedirs(output_subdir, exist_ok=True)
    
    # Save element info table
    element_info = create_element_info_table(elements, Te, Ne)
    info_path = os.path.join(output_subdir, 'element_info.csv')
    element_info.to_csv(info_path, index_label='element_index')
    print(f"Element info saved to: {info_path}")
    
    # Save weight spectra
    spectra_path = os.path.join(output_subdir, 'weight_spectra.npy')
    np.save(spectra_path, weight_spectra)
    print(f"Weight spectra saved to: {spectra_path}")
    
    # Save wavelength array
    wavelength_path = os.path.join(output_subdir, 'wavelength.npy')
    np.save(wavelength_path, wavelength)
    print(f"Wavelength array saved to: {wavelength_path}")
    
    # Save combined data as HDF5
    h5_path = os.path.join(output_subdir, 'element_weights.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_group('measurements')
        f.create_group('measurements/Measurement_1')
        f.create_group('measurements/Measurement_1/libs')
        f.create_group('measurements/Measurement_1/libs/metadata')
        f.create_group('measurements/Measurement_1/global_metadata')
        elements_encoded = np.array(elements, dtype='S10')
        f.create_dataset('measurements/Measurement_1/libs/metadata/elements', data=elements_encoded)
        f.create_dataset('measurements/Measurement_1/libs/data', data=weight_spectra)
        f.create_dataset('measurements/Measurement_1/libs/calibration', data=wavelength)
        f.create_dataset('measurements/Measurement_1/libs/metadata/Te', data=Te)
        f.create_dataset('measurements/Measurement_1/libs/metadata/Ne', data=Ne)
        f.create_dataset('measurements/Measurement_1/libs/metadata/concentration', data=CONCENTRATION)
        f.create_dataset('measurements/Measurement_1/libs/metadata/optical_path_length', data=OPTICAL_PATH_LENGTH)
        f.create_dataset('measurements/Measurement_1/libs/metadata/number_density', data=NUMBER_DENSITY)
        f.create_dataset('measurements/Measurement_1/libs/metadata/x', data=0)
        f.create_dataset('measurements/Measurement_1/libs/metadata/y', data=0)
        f.create_dataset('measurements/Measurement_1/libs/metadata/z', data=0)
        f.create_dataset('measurements/Measurement_1/libs/metadata/X_pos', data=0)
        f.create_dataset('measurements/Measurement_1/libs/metadata/Y_pos', data=0)
        f.create_dataset('measurements/Measurement_1/global_metadata/Width Spacing', data=0)
        f.create_dataset('measurements/Measurement_1/global_metadata/Height Spacing', data=0)
        
    print(f"Combined HDF5 file saved to: {h5_path}")


def save_multi_weights(weight_matrix: np.ndarray,
                        element_labels: list,
                        wavelength: np.ndarray,
                        te_labels: np.ndarray,
                        ne_labels: np.ndarray,
                        te_values: np.ndarray,
                        ne_values: np.ndarray,
                        output_dir: str = None):
    """
    Save multi TE/NE weight spectra to element_weights/multi_weights.h5
    
    This file is used by train_nn.py and predict_nn.py.
    
    Parameters
    ----------
    weight_matrix : np.ndarray
        Shape (n_combos * n_elements, n_wavelengths)
    element_labels : list
        Element name for each row
    wavelength : np.ndarray
        Wavelength calibration array
    te_labels : np.ndarray
        TE value for each row
    ne_labels : np.ndarray
        NE value for each row
    te_values : np.ndarray
        Array of TE values used in the grid
    ne_values : np.ndarray
        Array of NE values used in the grid
    output_dir : str, optional
        Output directory (defaults to script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    output_subdir = os.path.join(output_dir, 'element_weights')
    os.makedirs(output_subdir, exist_ok=True)
    
    # Derive unique elements (preserving order)
    unique_elements = list(dict.fromkeys(element_labels))
    
    h5_path = os.path.join(output_subdir, 'multi_weights.h5')
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('weight_matrix', data=weight_matrix)
        f.create_dataset('wavelength', data=wavelength)
        f.create_dataset('element_labels', data=np.array(element_labels, dtype='S10'))
        f.create_dataset('unique_elements', data=np.array(unique_elements, dtype='S10'))
        f.create_dataset('te_labels', data=te_labels)
        f.create_dataset('ne_labels', data=ne_labels)
        f.create_dataset('te_values', data=te_values)
        f.create_dataset('ne_values', data=ne_values)
    
    print(f"\nMulti-weight HDF5 saved to: {h5_path}")
    print(f"  Weight matrix shape: {weight_matrix.shape}")
    print(f"  Unique elements: {len(unique_elements)}")
    print(f"  TE/NE combinations: {len(te_values) * len(ne_values)}")
    
    # Also save a CSV summary
    csv_path = os.path.join(output_subdir, 'multi_weights_info.csv')
    info = pd.DataFrame({
        'element': element_labels,
        'Te': te_labels,
        'Ne': ne_labels
    })
    info.to_csv(csv_path, index_label='row_index')
    print(f"  Info CSV saved to: {csv_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Element Weight Spectra Generator")
    print("=" * 70)
    
    # 1. Display parameters
    print("\n1. Plasma Parameters:")
    print(f"   TE values ({len(TE)}): {TE[0]:.0f} - {TE[-1]:.0f} K")
    print(f"   NE values ({len(NE)}): {NE[0]:.2e} - {NE[-1]:.2e} cm^-3")
    print(f"   Total TE/NE combinations: {len(TE) * len(NE)}")
    print(f"   Element concentration: {CONCENTRATION*100:.0f}%")
    print(f"   Wavelength points: {len(wavelength)}")
    
    # 2. Get available elements from database
    print("\n2. Loading elements from database...")
    all_elements = get_elements_from_database(DATABASE_PATH)
    print(f"   Found {len(all_elements)} elements in database")
    
    # 3. Filter elements if specified
    elements = filter_elements(all_elements, INCLUDE_ELEMENTS)
    if INCLUDE_ELEMENTS:
        print(f"   Using specified subset: {len(elements)} elements")
    print(f"   Elements: {', '.join(elements)}")
    
    total = len(TE) * len(NE) * len(elements)
    print(f"\n   Total spectra to generate: {len(TE)} TE x {len(NE)} NE x {len(elements)} elements = {total}")
    
    # 4. Generate multi-weight spectra
    print("\n3. Generating multi-weight spectra...")
    weight_matrix, element_labels, te_labels, ne_labels = generate_multi_weights(
        elements=elements,
        wavelength=wavelength,
        te_values=TE,
        ne_values=NE,
        verbose=True
    )
    
    # 5. Save multi-weight results
    print("\n4. Saving multi-weight results...")
    save_multi_weights(
        weight_matrix=weight_matrix,
        element_labels=element_labels,
        wavelength=wavelength,
        te_labels=te_labels,
        ne_labels=ne_labels,
        te_values=TE,
        ne_values=NE
    )
    
    # 6. Also generate single TE/NE weights (using middle values) for backward compatibility
    mid_te = TE[len(TE) // 2]
    mid_ne = NE[len(NE) // 2]
    print(f"\n5. Generating single-weight spectra (TE={mid_te:.0f} K, NE={mid_ne:.2e})...")
    single_spectra, successful_elements, failed_elements = generate_element_weights(
        elements=elements,
        wavelength=wavelength,
        Te=mid_te,
        Ne=mid_ne,
        verbose=False
    )
    
    if failed_elements:
        print(f"   Failed elements: {list(failed_elements.keys())}")
    
    if len(successful_elements) > 0:
        save_single_weights(
            weight_spectra=single_spectra,
            elements=successful_elements,
            wavelength=wavelength,
            Te=mid_te,
            Ne=mid_ne
        )
    
    # 7. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Multi-weights: {weight_matrix.shape[0]} spectra ({len(TE)*len(NE)} combos x {len(elements)} elements)")
    print(f"  Single-weights: {len(successful_elements)} spectra (TE={mid_te:.0f}, NE={mid_ne:.2e})")
    print(f"  Wavelength points: {len(wavelength)}")
    print(f"  Output directory: element_weights/")
    print(f"  Files:")
    print(f"    - multi_weights.h5  (for train_nn.py / predict_nn.py)")
    print(f"    - element_weights.h5 (single TE/NE, backward compatible)")
    print("=" * 70)
