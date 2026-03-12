"""Generate and plot 5 random synthetic LIBS spectra."""

import random
from pathlib import Path

import plotly.graph_objects as go

from Sample_bootstrap import (
    DATABASE_PATH,
    NE_MAX,
    NE_MIN,
    SAMPLE_TYPES,
    TE_MAX,
    TE_MIN,
    SyntheticLIBSDataset,
    wavelength,
    data,
)


def pick_random_sample_types(n: int = 5) -> list[dict]:
    """Pick n random sample types and force one spectrum per type."""
    if not SAMPLE_TYPES:
        raise ValueError("SAMPLE_TYPES is empty.")

    n_pick = min(n, len(SAMPLE_TYPES))
    selected = random.sample(SAMPLE_TYPES, n_pick)

    # Ensure exactly one generated sample per selected sample type.
    return [
        {
            **sample_type,
            "n_samples": 1,
        }
        for sample_type in selected
    ]


def generate_dataset(sample_types: list[dict]) -> SyntheticLIBSDataset:
    """Generate synthetic dataset using the same pipeline as Sample_bootstrap.py."""
    return SyntheticLIBSDataset(
        sample_types=sample_types,
        wavelength=wavelength,
        db_path=DATABASE_PATH,
        te_range=(TE_MIN, TE_MAX),
        ne_range=(NE_MIN, NE_MAX),
        verbose=False,
    )


def plot_spectra(dataset: SyntheticLIBSDataset, output_html: Path) -> None:
    """Create interactive Plotly plot with sample ID and name."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; no spectra to plot.")

    table = dataset.sample_table.reset_index(drop=True)
    spectra = dataset.spectra

    fig = go.Figure()

    for i in range(len(table)):
        sample_id = table.loc[i, "sample_type_id"]
        sample_name = table.loc[i, "sample_type_name"]

        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=spectra[i],
                mode="lines",
                name=f"{sample_id} | {sample_name}",
                hovertemplate=(
                    "Sample ID: " + str(sample_id) + "<br>"
                    + "Sample Name: " + str(sample_name) + "<br>"
                    + "Wavelength: %{x:.3f} nm<br>"
                    + "Intensity: %{y:.6f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="5 Random Synthetic LIBS Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Intensity",
        template="plotly_white",
        legend_title="Sample",
        height=700,
    )

    fig.write_html(str(output_html), include_plotlyjs="cdn")


def main() -> None:
    output_html = Path(__file__).resolve().parent / "random_synthetic_spectra_plot.html"

    selected_types = pick_random_sample_types(n=5)
    dataset = generate_dataset(selected_types)
    plot_spectra(dataset, output_html)

    print(f"Generated {len(dataset)} spectra.")
    print("Selected samples:")
    for row in dataset.sample_table[["sample_type_id", "sample_type_name"]].drop_duplicates().itertuples(index=False):
        print(f"  - {row.sample_type_id} | {row.sample_type_name}")
    print(f"Plot saved to: {output_html}")
    print(f"Wavelength: {wavelength}")
    print(f"Data: {data}")

if __name__ == "__main__":
    main()
