"""
Weight Generator for Synthetic LIBS Spectra

This script generates synthetic LIBS (Laser-Induced Breakdown Spectroscopy) 
spectra for each element individually with concentration = 1 (100%).
These "weight spectra" can be used as basis functions for spectral analysis.

Each spectrum is generated with:
- Fixed concentration = 1 for that element
- User-configurable plasma temperature (Te)
- User-configurable electron number density (Ne)

The output is a matrix where:
- Each row represents one element's spectrum
- Each column represents a wavelength point
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

#read sample wavelength from h5 file
file_path = '/home/LIBS/prochazka/data/Running_projects/25_0069_3D_chemical_imaging/Measurements/mandible 266nm/mandible_266_v1.h5'
with h5py.File(file_path, 'r') as file:
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]

# ----------------------------------------------------------------------------
# Plasma Parameters (MODIFY THESE VALUES)
# ----------------------------------------------------------------------------
# Plasma temperature (Kelvin)
TE = 12705  # Default value, can be changed

# Electron number density (cm^-3)
NE = 1.79e18  # Default value, can be changed

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
# If specified, only these elements will be processed
# Example: INCLUDE_ELEMENTS = ['Fe', 'Si', 'Al', 'Ca', 'Mg']
INCLUDE_ELEMENTS = []  # Empty list = include all available elements


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


def generate_element_weights(elements: list,
                              wavelength: np.ndarray,
                              Te: float,
                              Ne: float,
                              n_density: float = NUMBER_DENSITY,
                              optical_path: float = OPTICAL_PATH_LENGTH,
                              concentration: float = CONCENTRATION,
                              verbose: bool = True) -> tuple:
    """
    Generate weight spectra for each element with concentration = 1.
    
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
    n_wavelengths = len(wavelength)
    
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


def create_element_info_table(elements: list, Te: float, Ne: float) -> pd.DataFrame:
    """
    Create a DataFrame with element information and plasma parameters.
    
    Parameters
    ----------
    elements : list
        List of element names
    Te : float
        Plasma temperature used
    Ne : float
        Electron number density used
        
    Returns
    -------
    pd.DataFrame
        DataFrame with element index, name, and plasma parameters
    """
    data = {
        'element': elements,
        'concentration': [CONCENTRATION] * len(elements),
        'Te': [Te] * len(elements),
        'Ne': [Ne] * len(elements),
    }
    return pd.DataFrame(data)


def save_results(weight_spectra: np.ndarray,
                 elements: list,
                 wavelength: np.ndarray,
                 Te: float,
                 Ne: float,
                 output_dir: str = None):
    """
    Save the weight spectra and metadata to files.
    
    Parameters
    ----------
    weight_spectra : np.ndarray
        Array of weight spectra (n_elements x n_wavelengths)
    elements : list
        List of element names
    wavelength : np.ndarray
        Wavelength array
    Te : float
        Plasma temperature used
    Ne : float
        Electron number density used
    output_dir : str, optional
        Output directory (defaults to script directory)
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    # Create output subdirectory
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
    
    # Save combined data as HDF5 for convenience
    try:
        import h5py
        h5_path = os.path.join(output_subdir, 'element_weights.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_group('measurements')
            f.create_group('measurements/Measurement_1')
            f.create_group('measurements/Measurement_1/libs')
            f.create_group('measurements/Measurement_1/libs/metadata')
            f.create_group('measurements/Measurement_1/global_metadata')
            # Save element names as fixed-length ASCII strings (compatible with all h5py versions)
            elements_encoded = np.array(elements, dtype='S10')  # Max 10 chars per element name
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
    except ImportError:
        print("h5py not available, skipping HDF5 output")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Element Weight Spectra Generator")
    print("=" * 70)
    
    # 1. Display plasma parameters
    print("\n1. Plasma Parameters:")
    print(f"   Temperature (Te): {TE:.0f} K")
    print(f"   Electron density (Ne): {NE:.2e} cm^-3")
    print(f"   Element concentration: {CONCENTRATION*100:.0f}%")
    
    # 2. Get available elements from database
    print("\n2. Loading elements from database...")
    all_elements = get_elements_from_database(DATABASE_PATH)
    print(f"   Found {len(all_elements)} elements in database")
    
    # 3. Filter elements if specified
    elements = filter_elements(all_elements, INCLUDE_ELEMENTS)
    if INCLUDE_ELEMENTS:
        print(f"   Using specified subset: {len(elements)} elements")
    print(f"   Elements: {', '.join(elements)}")
    
    # 4. Create wavelength array
    print("\n3. Creating wavelength array...")
    
    # 5. Generate weight spectra
    print("\n4. Generating weight spectra for each element...")
    weight_spectra, successful_elements, failed_elements = generate_element_weights(
        elements=elements,
        wavelength=wavelength,
        Te=TE,
        Ne=NE,
        verbose=True
    )
    
    # 6. Report failed elements
    if failed_elements:
        print("\n   Failed elements:")
        for elem, error in failed_elements.items():
            print(f"      {elem}: {error}")
    
    # 7. Save results
    if len(successful_elements) > 0:
        print("\n5. Saving results...")
        save_results(
            weight_spectra=weight_spectra,
            elements=successful_elements,
            wavelength=wavelength,
            Te=TE,
            Ne=NE
        )
    else:
        print("\n5. No spectra generated, skipping save.")
    
    print("\n" + "=" * 70)
    print(f"Done! Generated weight spectra for {len(successful_elements)} elements")
    print(f"Plasma parameters: Te = {TE:.0f} K, Ne = {NE:.2e} cm^-3")
    print("=" * 70)
    
    # Display summary
    print("\nSummary:")
    print(f"  - Successful elements: {len(successful_elements)}")
    print(f"  - Failed elements: {len(failed_elements)}")
    if successful_elements:
        print(f"  - Spectrum shape: {weight_spectra.shape}")
        print(f"  - Output directory: element_weights/")
