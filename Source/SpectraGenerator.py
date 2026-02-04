#Function generates optical emission spectra based on given elemnt, temperature and pressure and electron density
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from LIBSmethods import voigt, partition_function
import h5py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
"""def generate_spectra(element, temperature, electron_density):
    #Calculate the energy of the electron in the ground state
    energy_ground_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the excited state
    energy_excited_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the ionized state
    energy_ionized_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
    #Calculate the energy of the electron in the metastable state
    energy_metastable_state = const.Rydberg * const.Z**2 / (1 + const.Z**2)
"""

#Constants========================================================
kb = const.k*10**7 #erg/K
h = const.h*10**7 #erg*s
c = const.c #m/s
e = const.e #C
me = const.electron_mass*1000 #g
#-----------------------------------------------------------------
Te = 12705 #K
Ne = 1.79e+18 #cm^-3
l = 1.4e-04
N=1e-4 #cm^-3
C= 1

print(f"kb: {kb}")
print(f"h: {h}")
print(f"c: {c}")
print(f"e: {e}")
print(f"me: {me}")

#read sample wavelengths

#file_path = '/home/LIBS/prochazka/data/Running_projects/25_0069_3D_chemical_imaging/Measurements/mandible 266nm/mandible_266_v1.h5'
#with h5py.File(file_path, 'r') as file:
#    wavelength = file['measurements/Measurement_1/libs/calibration'][:]

#element = 'Li'

def create_spectra(element, wavelength, Te=12705, Ne=1.79e+18, N=1e-4, C=1, l=1.4e-04):
    """
    Generate synthetic optical emission spectra for a given element.
    
    Parameters:
    -----------
    element : str
        Element symbol (e.g., 'Li', 'Cu')
    wavelength : array-like
        Wavelength array (in nm) where the spectrum will be evaluated
    Te : float
        Temperature in Kelvin (default: 12705)
    Ne : float
        Electron number density in cm^-3 (default: 1.79e+18)
    N : float
        Number density in cm^-3 (default: 1e-4)
    C : float
        Content of element (default: 1 ~ 100%)
    l : float
        Optical path length in cm (default: 1.4e-04)
    
    Returns:
    --------
    Ifin_voigt : numpy array
        Synthetic spectrum intensity as a function of wavelength
    """
    # Single database file containing all tables
    DATABASE_PATH = '/home/LIBS/prochazka/data/Running_projects/24_0057_LIBSdata_processing/Methods/Mapping/Java/LIBS_data.db'

    # Use context manager to ensure connection is always closed
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()  
        # Get the partition function for the element elem
        cursor.execute("SELECT Elem_name, ion_state, Wavelength, Ei, Ek, gi, gk, Ak FROM QuantParam WHERE Elem_name = ?", (element,))
        QuantParam = pd.DataFrame(cursor.fetchall(), columns=['Elem_name', 'ion_state', 'Wavelength', 'Ei', 'Ek', 'gi', 'gk', 'Ak'])
        print(f"QuantParam: {QuantParam.head()}")
        
        cursor.execute("SELECT Eion FROM E_ion WHERE Elem_name = ?", (element+'+I',))
        E_ion_result = cursor.fetchall()
        if not E_ion_result:
            raise ValueError(f"Ionization energy (E_ion) not found for element '{element}+I' in database. Please ensure the element data exists in the E_ion table.")
        E_ion = E_ion_result[0][0]
        print(f"E_ion: {E_ion}")

    PF_I, PF_II = partition_function(element, Te)
    S10 = (((2*PF_II)/(Ne*PF_I))*((me*kb*Te)/((h**2)/(2*np.pi)))**(1.5))*np.exp(-(E_ion*1.60217e-12)/(kb*Te)) if not QuantParam.empty else 1
    kt = ((QuantParam['Wavelength']**4)/(8*np.pi*c)) * (QuantParam['Ak']*QuantParam['gk']*np.exp(-(QuantParam['Ei']*1.60217e-12)/(kb*Te))) * (1-np.exp(-1.60217e-12*(QuantParam['Ek']-QuantParam['Ei'])/(kb*Te))) / np.where(QuantParam['ion_state']=="I", PF_I, PF_II)
    ri = np.where(QuantParam['ion_state']=="I", 1/(1+S10), S10/(1+S10))
    ion_state = np.where(QuantParam['ion_state']=="I", 1, 0)
    Lp = ((8*np.pi*h*c)/(10*QuantParam['Wavelength']**3))*N*np.exp((-1.60217e-12*(QuantParam['Ek']-QuantParam['Ei']))/(kb*Te))*(QuantParam['gk']/QuantParam['gi'])
    tau = C*N*ri*l*kt
    Ifin = Lp*(1-np.exp(-tau))

    # Apply Voigt profile to each spectral line and sum them
    # Convert to numpy arrays to avoid pandas broadcasting issues
    wavelength_arr = np.array(wavelength)
    wavelength_lines = QuantParam['Wavelength'].values  # Convert pandas Series to numpy array
    Ifin_arr = Ifin.values  # Convert pandas Series to numpy array

    # Initialize the spectrum array
    Ifin_voigt = np.zeros_like(wavelength_arr)

    # Sum Voigt profiles for each spectral line
    gamma_fit = 0.1
    sigma_fit = 0.006
    for i in range(len(wavelength_lines)):
        Ifin_voigt += voigt(wavelength_arr, wavelength_lines[i], Ifin_arr[i], gamma_fit, sigma_fit)

    return Ifin_voigt

#Ifin_voigt = create_spectra(element, wavelength, Te, Ne, N, C, l)
#Plot Ifin_voigt as a function of wavelength using plotly
#import plotly.graph_objects as go
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=wavelength, y=Ifin_voigt, mode='lines', name=f'Spectrum: {element}'))
#fig.update_layout(title=f'Spectrum: {element}', xaxis_title='Wavelength (nm)', yaxis_title='Intensity (a.u.)')
#fig.write_html('SpectraGenerator.html')
#print("Plot saved to SpectraGenerator.html")