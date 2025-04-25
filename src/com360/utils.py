"""Utility functions and constants for DMCA data analysis."""

from pathlib import Path
from time import perf_counter

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rcParams

from com360.logging_config import get_logger

log = get_logger(__name__)

DEFAULT_DATA_DIR = Path("data") / "google-websearch-copyright-removals"
DEFAULT_BASE_PLOT_DIR = Path("copyright_analysis_plots")

PREFERRED_FONTS = [
    "NanumGothic",
    "Noto Sans CJK KR",
    "Malgun Gothic",
    "Apple SD Gothic Neo",
    "Arial Unicode MS",
    "DejaVu Sans",
]

DEFAULT_PLOT_STYLE = {
    "figure.figsize": (10, 6),
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "figure.autolayout": True,
    "font.size": 10.0,
    "axes.titlesize": "large",
    "axes.labelsize": "medium",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "axes.facecolor": "#EEEEEE",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.dpi": 150,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}


def find_available_font(font_list: list[str]) -> str | None:
    """Find the first available font from a list."""
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in font_list:
        if font_name in available_fonts:
            log.debug("Found available font.", font_name=font_name)
            return font_name
    log.warning(
        "No preferred fonts found. Falling back to default.",
        preferred_fonts=font_list,
    )
    return None


def apply_plot_styling(style_dict: dict | None = None) -> None:
    """Apply global Matplotlib styling settings.

    Sets the primary font and updates rcParams with provided style dictionary.

    Parameters
    ----------
    style_dict : dict | None, optional
        A dictionary of rcParams settings to apply.
        Defaults to DEFAULT_PLOT_STYLE.
    """
    log.info("Applying global Matplotlib plot styling...")

    target_font = find_available_font(PREFERRED_FONTS)
    if target_font:
        log.info("Setting primary font.", font_name=target_font)
        rcParams["font.family"] = target_font
        rcParams["axes.unicode_minus"] = False
    else:
        log.error(
            "Could not find suitable font. Plots may have missing characters."
        )

    style_to_apply = (
        style_dict if style_dict is not None else DEFAULT_PLOT_STYLE
    )
    try:
        rcParams.update(style_to_apply)
        log.info(
            "Successfully applied plot style settings.",
            count=len(style_to_apply),
        )
    except Exception:
        log.exception("Failed to apply some plot style settings.")


def load_parquet_data(
    file_path: Path, columns: list[str] | None = None
) -> pl.DataFrame | None:
    """Load specified columns from a Parquet file into a Polars DataFrame.

    Parameters
    ----------
    file_path : Path
        Specify the path to the Parquet file. Should be an absolute path
        or relative to the current working directory.
    columns : list[str] | None, optional
        Specify the list of columns to load. Load all if None. Defaults to None.

    Returns
    -------
    pl.DataFrame | None
        Return the loaded Polars DataFrame, or None if loading fails.
    """
    log.info(
        "Attempting to load Parquet data.",
        path=str(file_path),
        columns=columns or "all",
    )
    start_time = perf_counter()
    try:
        if not file_path.is_file():
            log.error("Parquet file not found.", path=str(file_path))
            return None

        df = pl.read_parquet(file_path, columns=columns)
        load_time = perf_counter() - start_time
        log.info(
            "Successfully loaded Parquet data.",
            path=str(file_path),
            rows=len(df),
            duration_sec=round(load_time, 2),
        )
        return df
    except pl.ComputeError as e:
        log.error(
            "Polars compute error loading Parquet data.",
            path=str(file_path),
            error=str(e),
        )
        return None
    except Exception:
        log.exception(
            "Unexpected error loading Parquet data.", path=str(file_path)
        )
        return None


def save_plot(fig: plt.Figure, save_path: Path) -> None:
    """Save a Matplotlib figure to a specified path, ensuring directory exists.

    Parameters
    ----------
    fig : plt.Figure
        The Matplotlib Figure object to save.
    save_path : Path
        The full path (including filename and extension) where the plot should be saved.
        Should be an absolute path or relative to the current working directory.
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        log.info("Saved plot.", path=str(save_path))
    except Exception:
        log.exception("Failed to save plot.", path=str(save_path))
