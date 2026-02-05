import os
import numpy as np
import h5py

weight_spectra = np.array([1, 2, 3, 4, 5])
wavelength = np.array([5, 6, 7, 8, 9])
elements = np.array(['Fe', 'Si', 'Al', 'Ca', 'Mg'])
Te = 12705
Ne = 1.79e18
CONCENTRATION = 1
OPTICAL_PATH_LENGTH = 1.4e-04
NUMBER_DENSITY = 1e-4

h5_path = os.path.join('test_temporary.h5')
with h5py.File(h5_path, 'w') as f:
    f.create_group('measurements')
    f.create_group('measurements/Measurement_1')
    f.create_group('measurements/Measurement_1/libs')
    f.create_group('measurements/Measurement_1/libs/metadata')

    f.create_dataset('measurements/Measurement_1/libs/data', data=weight_spectra)
    f.create_dataset('measurements/Measurement_1/libs/calibration', data=wavelength)
    # Encode strings as bytes for HDF5 compatibility
    elements_encoded = np.array(elements, dtype='S10')
    f.create_dataset('measurements/Measurement_1/libs/metadata/elements', data=elements_encoded)
    print(f"Combined HDF5 file saved to: {h5_path}")

def print_hdf5_tree(name, obj, prefix="", is_last=True):
    """Print HDF5 structure as a tree"""
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{name.split('/')[-1]}")
    
    if isinstance(obj, h5py.Group):
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            is_last_item = (i == len(items) - 1)
            extension = "    " if is_last else "│   "
            print_hdf5_tree(f"{name}/{key}", value, prefix + extension, is_last_item)

with h5py.File(h5_path, 'r') as f:
    print("HDF5 file structure")
    print("=" * 50)
    for key in f.keys():
        print_hdf5_tree(key, f[key], "", True)
    print("=" * 50)