"""
Script to read and plot element weight spectra from element_weights.h5
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Path to the HDF5 file
h5_path = os.path.join(os.path.dirname(__file__), 'element_weights', 'element_weights.h5')

# Read data from HDF5 file
with h5py.File(h5_path, 'r') as f:
    weight_spectra = f['measurements/Measurement_1/libs/data'][:]
    wavelength = f['measurements/Measurement_1/libs/calibration'][:]
    elements = [e.decode('utf-8') for e in f['measurements/Measurement_1/libs/metadata/elements'][:]]
    Te = f['measurements/Measurement_1/libs/metadata/Te'][()]
    Ne = f['measurements/Measurement_1/libs/metadata/Ne'][()]
    
    print(f"Loaded {len(elements)} element spectra")
    print(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
    print(f"Spectra shape: {weight_spectra.shape}")
    print(f"Plasma parameters: Te = {Te:.0f} K, Ne = {Ne:.2e} cm^-3")
    print(f"Elements: {', '.join(elements)}")

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot each element's spectrum with different colors
colors = plt.cm.tab20(np.linspace(0, 1, len(elements)))

for i, (elem, spectrum) in enumerate(zip(elements, weight_spectra)):
    # Only plot if spectrum has non-zero values
    if np.max(spectrum) > 0:
        ax.plot(wavelength, spectrum, label=elem, color=colors[i], alpha=0.8, linewidth=0.5)

ax.set_xlabel('Wavelength (nm)', fontsize=12)
ax.set_ylabel('Intensity (a.u.)', fontsize=12)
ax.set_title(f'Element Weight Spectra (Te = {Te:.0f} K, Ne = {Ne:.2e} cm⁻³)', fontsize=14)
ax.legend(loc='upper right', ncol=4, fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('element_weights_plot.png', dpi=150)
print("\nPlot saved to: element_weights_plot.png")

# Also create an interactive plot with plotly
try:
    import plotly.graph_objects as go
    
    fig_plotly = go.Figure()
    
    for i, (elem, spectrum) in enumerate(zip(elements, weight_spectra)):
        if np.max(spectrum) > 0:
            fig_plotly.add_trace(go.Scatter(
                x=wavelength, 
                y=spectrum, 
                mode='lines', 
                name=elem,
                line=dict(width=1)
            ))
    
    fig_plotly.update_layout(
        title=f'Element Weight Spectra (Te = {Te:.0f} K, Ne = {Ne:.2e} cm⁻³)',
        xaxis_title='Wavelength (nm)',
        yaxis_title='Intensity (a.u.)',
        hovermode='x unified',
        legend=dict(font=dict(size=10))
    )
    
    fig_plotly.write_html('element_weights_plot.html')
    print("Interactive plot saved to: element_weights_plot.html")
except ImportError:
    print("Plotly not available, skipping interactive plot")

# Show the matplotlib plot
plt.show()
