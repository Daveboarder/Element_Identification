"""Benchmark: sequential vs parallel spectra generation."""
import time
import numpy as np
import pandas as pd
import Sample_bootstrap as sb
from Sample_bootstrap import (
    generate_sample_table, validate_elements, generate_synthetic_spectra
)

sample_types = sb.SAMPLE_TYPES[:10]
tables = []
for i, st in enumerate(sample_types):
    try:
        validate_elements(st["concentration_ranges"], sb.DATABASE_PATH)
    except ValueError:
        continue
    tables.append(
        generate_sample_table(
            concentration_ranges=st["concentration_ranges"],
            n_samples=st["n_samples"],
            sample_id=st["sample_id"],
            sample_name=st["sample_name"],
            te_range=(sb.TE_MIN, sb.TE_MAX),
            ne_range=(sb.NE_MIN, sb.NE_MAX),
            random_seed=42 + i,
        )
    )
combined = pd.concat(tables, ignore_index=True).fillna(0)
print(f"Combined table: {len(combined)} samples\n", flush=True)

# Sequential
t0 = time.perf_counter()
s1 = generate_synthetic_spectra(
    combined, sb.wavelength, db_path=sb.DATABASE_PATH, verbose=False, n_workers=1
)
seq = time.perf_counter() - t0
print(f"Sequential (1 worker):   {seq:.2f}s  ({seq / len(combined) * 1000:.1f} ms/sample)", flush=True)

# Parallel
for nw in [4, 16]:
    t0 = time.perf_counter()
    s2 = generate_synthetic_spectra(
        combined, sb.wavelength, db_path=sb.DATABASE_PATH, verbose=False, n_workers=nw
    )
    par = time.perf_counter() - t0
    diff = np.abs(s1 - s2).max()
    print(
        f"Parallel ({nw:2d} workers):   {par:.2f}s  "
        f"({par / len(combined) * 1000:.1f} ms/sample)  "
        f"speedup={seq / par:.1f}x  maxdiff={diff:.2e}",
        flush=True,
    )
