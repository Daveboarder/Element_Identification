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

h5_path = '/mnt/data/projects/Running_projects/24_0053_CM24_111/Data/Li_some random data/2022_09_20_Buday_Li Kal'
#read all h5 files in the folder
#os.path.join('test_temporary.h5')
h5_files = [f for f in os.listdir(h5_path) if f.endswith('.h5')]
for file in h5_files:
    with h5py.File(os.path.join(h5_path, file), 'r') as f:
        print(f.keys())

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

# Check structure of first h5 file in the h5_path
element_weights_path = os.path.join(h5_path, h5_files[0])
#element_weights_path = os.path.join(os.path.dirname(__file__), '..', 'element_weights', 'element_weights.h5')
with h5py.File(element_weights_path, 'r') as f:
    print(f"\nHDF5 file structure: {element_weights_path}")
    print("=" * 50)
    for key in f.keys():
        print_hdf5_tree(key, f[key], "", True)
    print("=" * 50)
    # Also print attributes
    print("\nFile attributes:")
    for attr_name, attr_value in f.attrs.items():
        print(f"  {attr_name}: {attr_value}")