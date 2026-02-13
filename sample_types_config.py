"""
Sample Type Definitions for Synthetic LIBS Spectra Generator

Define multiple sample types, each with:
- 'sample_id': Unique identifier for the sample type
- 'sample_name': Human-readable name for the sample type
- 'n_samples': Number of samples to generate for this type
- 'concentration_ranges': Dict of element concentration ranges {element: (min, max)}

Format for concentration_ranges: 'Element': (min_concentration, max_concentration)
Concentrations are given as fractions (0 = 0%, 1 = 100%)

IMPORTANT: Only elements specified will have non-zero concentrations.
           All other elements will have concentration = 0.
"""

SAMPLE_TYPES = [
    {
        'sample_id': 'MINERAL_001',
        'sample_name': 'Monazite',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Monazite
            'Ce': (0.24, 0.31),
            'La': (0.1445, 0.187),
            'Nd': (0.051, 0.119),
            'P': (0.5, 0.62),
            'O': (0.1, 0.3),
            # Trace elements
            'Pr': (0.0, 0.009),
            'Sm': (0.0, 0.005),
            'Y': (0.0, 0.0032),
        }
    },
    {
        'sample_id': 'MINERAL_002',
        'sample_name': 'Bastnaesite',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'Ce': (0.41, 0.46),
            'La': (0.28, 0.31),
            'C': (0.2, 0.3),
            'O': (0.1, 0.2),
            'F': (0.2, 0.3),
            # Trace elements
            'Sm': (0.0, 0.001),
            'Nd': (0.1, 0.12),
            'Pr': (0.034, 0.0425),
        }
    },
    {
        'sample_id': 'MINERAL_003',
        'sample_name': 'Xenotime',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for minerals samples
            'Y': (0.39, 0.52),
            'P': (0.2, 0.25),
            'O': (0.01, 0.015),
            # Trace elements
            'Dy': (0.034, 0.051),
            'Er': (0.01, 0.012),
            'Gd': (0.0085, 0.01),
            'Tb': (0.004, 0.00642),
            'Yb': (0.01, 0.012),
            'Lu': (0.0002, 0.00025),
            'Ho': (0.001, 0.0015),
        }
    },
    {
        'sample_id': 'MINERAL_004',
        'sample_name': 'Parisite',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for minerals samples
            'Ce': (0.24, 0.31),
            'La': (0.1445, 0.187),
            'Nd': (0.051, 0.119),
            'F': (0.5, 0.62),
            'O': (0.1, 0.3),
            'C': (0.2, 0.3),
            'Ca': (0.045, 0.055),
            # Trace elements
            'Pr': (0.008, 0.009),
        }
    },
    {
        'sample_id': 'MINERAL_005',
        'sample_name': 'Obsidian_glass',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for biological samples
            'Na': (0.005, 0.007),
            'Mg': (0.012, 0.015),
            'Al': (0.08, 0.09),
            'Si': (0.25, 0.3),
            'K': (0.05, 0.07),
            'Ca': (0.045, 0.055),
            'Fe': (0.025, 0.032),
            'Ti': (0.004, 0.006),
            'O': (0.05, 0.06),
            # Trace elements
            'Cr': (0.0, 0.0001),
            'Ce': (0.0006, 0.0008),
            'Y': (0.0002, 0.0005),
            'Nd': (0.0003, 0.0006),
            'Pr': (0.0001, 0.0002),
            'Gd': (0.00004, 0.00007),
            'Dy': (0.00002, 0.00004),
            'Er': (0.00001, 0.00002),
            'Yb': (0.000005, 0.000008),
            'Lu': (0.000002, 0.000005),
        }
    },
    {
        'sample_id': 'MINERAL_006',
        'sample_name': 'Pure_Ce',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'Ce': (0.85, 1.0),
            'O': (0.1, 0.15),
        }
    },
    {
        'sample_id': 'MINERAL_007',
        'sample_name': 'Pure_La',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'La': (0.85, 1.0),
            'O': (0.1, 0.15),
        }
    },
    {
        'sample_id': 'MINERAL_008',
        'sample_name': 'Pure_Nd',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'Nd': (0.85, 1.0),
            'O': (0.1, 0.15),
        }
    },
    {
        'sample_id': 'MINERAL_009',
        'sample_name': 'Pure_Pr',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'Pr': (0.85, 1.0),
            'O': (0.1, 0.15),
        }
    },
    {
        'sample_id': 'MINERAL_010',
        'sample_name': 'Pure_Sm',
        'n_samples': 100,
        'concentration_ranges': {
            # Major elements for Bastnaesite (Ce-fluorocarbonate)
            'Sm': (0.85, 1.0),
            'O': (0.1, 0.15),
        }
    },
]
