#plot synthetic spectra from synthetic_data.h5 using plotly
import plotly.graph_objects as go
import h5py
import numpy as np
from Sample_bootstrap import unit_norm
from LIBSmethods import movingMinimum
#read synthetic_data.h5
with h5py.File('synthetic_data.h5', 'r') as file:
    spectra = file['measurements/Measurement_1/libs/data'][:]
    wavelength = file['measurements/Measurement_1/libs/calibration'][:]
    sample_name = file['measurements/Measurement_1/libs/metadata/samples/sample_type_name'][:]
    sample_id = file['measurements/Measurement_1/libs/metadata/samples/sample_type_id'][:]
    # Decode bytes to strings for sample names
    sample = [sample_name[i].decode('utf-8') if isinstance(sample_name[i], bytes) else str(sample_name[i]) for i in range(len(sample_name))]

with h5py.File('/mnt/data/projects/Running_projects/26_0128_Element_Identification/Data/FG_OBSIDIAN.h5', 'r') as file:
    data = file['measurements/Measurement_1/libs/data'][:]

obsidian = np.mean(data, axis=0)
obsidian = unit_norm(obsidian)
obsidian = movingMinimum(obsidian)
print(spectra.shape)
unique_id = np.unique(sample_id)
print(unique_id.shape)
print(unique_id)
#aggregate spectra by sample_id
spectra_aggregated = np.zeros((len(unique_id), len(wavelength)))
for i in range(len(unique_id)):
    spectra_aggregated[i,:] = np.mean(spectra[sample_id == unique_id[i],:], axis=0)
    spectra_aggregated[i,:] = unit_norm(spectra_aggregated[i,:])
spectra_aggregated = np.vstack((spectra_aggregated, obsidian))
unique_id = np.append(unique_id, 'OBSIDIAN')
fig = go.Figure()
for i in range(len(unique_id)):
    # Decode unique_id to string for Plotly
    name = unique_id[i].decode('utf-8') if isinstance(unique_id[i], bytes) else str(unique_id[i])
    fig.add_trace(go.Scatter(x=wavelength, y=spectra_aggregated[i,:], mode='lines', name=name))
fig.show()