"""Observer-Window Life analysis package."""

from owl.analysis.compare_runs import (
    compare_conditions,
    load_metric_tables,
    parameter_sweep_heatmap,
)
from owl.analysis.plots import (
    make_animation_from_zarr,
    plot_global_integration,
    plot_population_by_trait,
    plot_signal_channel_totals,
)
from owl.analysis.zarr_reader import load_field, open_run_zarr

__all__ = [
    "open_run_zarr",
    "load_field",
    "plot_global_integration",
    "plot_population_by_trait",
    "plot_signal_channel_totals",
    "make_animation_from_zarr",
    "load_metric_tables",
    "compare_conditions",
    "parameter_sweep_heatmap",
]
