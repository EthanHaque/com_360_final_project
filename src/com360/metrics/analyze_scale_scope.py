"""Analyze the scale and scope of DMCA takedown requests."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from com360.logging_config import get_logger, setup_logging
from com360.utils import (
    DEFAULT_BASE_PLOT_DIR,
    DEFAULT_DATA_DIR,
    apply_plot_styling,
    load_parquet_data,
    save_plot,
)

MODULE_PLOT_DIR = DEFAULT_BASE_PLOT_DIR / "scale_scope"
REQUESTS_FILENAME = "requests.parquet"

DATE_COL = "Date"
REPORTING_ORG_COL = "Reporting organization name"
COPYRIGHT_OWNER_COL = "Copyright owner name"
REQUEST_ID_COL = "Request ID"
MONTH_COL = "Month"
REQUEST_COUNT_COL = "request_count"

log = get_logger(__name__)


def _load_and_prepare_requests_data(
    requests_path: Path,
) -> tuple[pl.DataFrame | None, str | None]:
    """Load requests data and determine required columns including optional ID.

    Parameters
    ----------
    requests_path : Path
        Specify the path to the requests Parquet file.

    Returns
    -------
    tuple[pl.DataFrame | None, str | None]
        Return a tuple containing the loaded DataFrame with selected columns (or None on failure),
        and the name of the actual request ID column found (or None).
    """
    log.info("Loading and preparing requests data...")
    required_cols = [
        DATE_COL,
        REPORTING_ORG_COL,
        COPYRIGHT_OWNER_COL,
    ]
    potential_request_id_col = REQUEST_ID_COL

    requests_df_full = load_parquet_data(requests_path)
    if requests_df_full is None:
        log.error("Failed to load full request data.")
        return None, None

    actual_request_id_col = None
    if potential_request_id_col in requests_df_full.columns:
        actual_request_id_col = potential_request_id_col
        required_cols.append(actual_request_id_col)
        log.info(
            "Found request ID column for distinct counting.",
            col=actual_request_id_col,
        )
    else:
        log.warning(
            f"Column '{potential_request_id_col}' not found. Counting rows instead of distinct requests."
        )

    try:
        requests_df_selected = requests_df_full.select(required_cols)
        log.info("Selected required columns.", columns=required_cols)
        return requests_df_selected, actual_request_id_col
    except pl.ColumnNotFoundError as e:
        log.error(
            "Required column not found during selection.",
            error=str(e),
            required=required_cols,
        )
        return None, None
    except Exception:
        log.exception("Error selecting required columns from DataFrame.")
        return None, None


def plot_top_n_by_request_count(
    df: pl.DataFrame,
    group_col: str,
    request_id_col: str | None = None,
    output_dir: Path = MODULE_PLOT_DIR,
    top_n: int = 20,
    plot_filename_base: str = "top_items",
    title_prefix: str = "Top",
) -> None:
    """Generate and save a bar chart of top N items by request count.

    Count distinct requests if `request_id_col` is provided, otherwise count rows.

    Parameters
    ----------
    df : pl.DataFrame
        Provide the Polars DataFrame containing the data. Needs `group_col` and optionally `request_id_col`.
    group_col : str
        Specify the column name to group by (e.g., 'Reporting organization name').
    request_id_col : str | None, optional
        Define the column name of the unique request identifier. If None, count rows per group.
        Defaults to None.
    output_dir : Path, optional
        Set the directory to save the plot image. Defaults to MODULE_PLOT_DIR.
    top_n : int, optional
        Determine the number of top items to display. Defaults to 20.
    plot_filename_base : str, optional
        Set the base name for the output PNG file. Defaults to "top_items".
    title_prefix : str, optional
        Define the prefix for the plot title. Defaults to "Top".
    """
    log.info(
        f"Generating top {top_n} {group_col} plot...",
        top_n=top_n,
        group_col=group_col,
    )
    try:
        if group_col not in df.columns:
            log.error(
                f"Grouping column '{group_col}' not found in DataFrame.",
                columns=df.columns,
            )
            return
        if request_id_col and request_id_col not in df.columns:
            log.warning(
                f"Request ID column '{request_id_col}' provided but not found in DataFrame slice. Counting rows.",
                columns=df.columns,
            )
            request_id_col = None

        if request_id_col:
            log.debug(
                "Counting distinct requests using column.",
                request_col=request_id_col,
            )
            agg_expr = pl.n_unique(request_id_col)
        else:
            log.debug("Counting rows per group.")
            agg_expr = pl.count()

        top_items_pl = (
            df.group_by(group_col)
            .agg(agg_expr.alias(REQUEST_COUNT_COL))
            .sort(REQUEST_COUNT_COL, descending=True)
            .head(top_n)
        )

        if top_items_pl.is_empty():
            log.warning(
                "No data found after aggregation to plot.", group_col=group_col
            )
            return

        top_items_pd = top_items_pl.to_pandas().set_index(group_col)[
            REQUEST_COUNT_COL
        ]

        if top_items_pd.empty:
            log.warning(
                "Pandas series is empty, cannot plot.", group_col=group_col
            )
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        top_items_pd.sort_values(ascending=True).plot(kind="barh", ax=ax)

        plot_title = f"{title_prefix} {top_n} {group_col.replace('_', ' ').title()} by Number of Requests"
        ax.set_title(plot_title)
        ax.set_xlabel("Number of Requests")
        ax.set_ylabel(group_col.replace("_", " ").title())

        save_path = output_dir / f"{plot_filename_base}_by_requests.png"
        save_plot(fig, save_path)
        plt.close(fig)

    except Exception:
        log.exception(f"Failed to generate top {group_col} plot.")


def plot_monthly_requests(
    requests_df: pl.DataFrame,
    date_col: str = DATE_COL,
    output_dir: Path = MODULE_PLOT_DIR,
) -> None:
    """Generate and save a time series plot of monthly requests.

    Parameters
    ----------
    requests_df : pl.DataFrame
        Provide the Polars DataFrame containing request data with a date column.
    date_col : str, optional
        Specify the name of the column containing datetime information. Defaults to DATE_COL.
    output_dir : Path, optional
        Set the directory to save the plot image. Defaults to MODULE_PLOT_DIR.
    """
    log.info(
        "Generating monthly requests time series plot...", date_col=date_col
    )
    try:
        if date_col not in requests_df.columns:
            log.error(
                f"Date column '{date_col}' not found in DataFrame.",
                columns=requests_df.columns,
            )
            return
        if not isinstance(requests_df[date_col].dtype, pl.Datetime):
            log.error(
                f"Date column '{date_col}' is not Datetime type.",
                dtype=requests_df[date_col].dtype,
            )
            log.warning(
                "Date column type is not Datetime. Ensure it was parsed correctly during data loading."
            )
            return

        monthly_counts_pl = (
            requests_df.group_by(
                pl.col(date_col).dt.truncate("1mo").alias(MONTH_COL)
            )
            .agg(pl.count().alias(REQUEST_COUNT_COL))
            .sort(MONTH_COL)
        )

        if monthly_counts_pl.is_empty():
            log.warning("No monthly request data found to plot.")
            return

        monthly_counts_pd = monthly_counts_pl.to_pandas().set_index(MONTH_COL)[
            REQUEST_COUNT_COL
        ]

        if monthly_counts_pd.empty:
            log.warning("Monthly counts Pandas series is empty, cannot plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        monthly_counts_pd.plot(kind="line", marker=".", linestyle="-", ax=ax)

        ax.set_title("Monthly Copyright Takedown Requests Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Number of Requests")

        save_path = output_dir / "monthly_requests_timeseries.png"
        save_plot(fig, save_path)
        plt.close(fig)

    except Exception:
        log.exception("Failed to generate monthly requests plot.")


def _run_scale_scope_plots(
    requests_df: pl.DataFrame,
    actual_request_id_col: str | None,
    output_dir: Path,
) -> None:
    """Execute all plotting functions for the scale/scope analysis.

    Parameters
    ----------
    requests_df : pl.DataFrame
        The prepared DataFrame containing necessary columns.
    actual_request_id_col : str | None
        The name of the request ID column if found, otherwise None.
    output_dir : Path
        The directory to save the plots.
    """
    log.info("Running scale and scope plot generation...")

    # 1. Temporal Trends
    plot_monthly_requests(requests_df.select(DATE_COL), output_dir=output_dir)

    # 2. Top Senders (Reporting Orgs)
    sender_cols = [REPORTING_ORG_COL]
    if actual_request_id_col:
        sender_cols.append(actual_request_id_col)
    plot_top_n_by_request_count(
        df=requests_df.select(sender_cols),
        group_col=REPORTING_ORG_COL,
        request_id_col=actual_request_id_col,
        output_dir=output_dir,
        top_n=25,
        plot_filename_base="top_reporting_orgs",
        title_prefix="Top",
    )

    # 3. Top Senders (Copyright Owners)
    owner_cols = [COPYRIGHT_OWNER_COL]
    if actual_request_id_col:
        owner_cols.append(actual_request_id_col)
    plot_top_n_by_request_count(
        df=requests_df.select(owner_cols),
        group_col=COPYRIGHT_OWNER_COL,
        request_id_col=actual_request_id_col,
        output_dir=output_dir,
        top_n=25,
        plot_filename_base="top_copyright_owners",
        title_prefix="Top",
    )


def main() -> None:
    """Load data, prepare it, and run scale/scope analyses."""
    log.info("Starting Scale and Scope Analysis.")

    apply_plot_styling()
    MODULE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Plot output directory.", path=str(MODULE_PLOT_DIR))

    current_working_directory = Path.cwd()
    requests_path = (
        current_working_directory / DEFAULT_DATA_DIR / REQUESTS_FILENAME
    )

    requests_df, actual_request_id_col = _load_and_prepare_requests_data(
        requests_path
    )
    if requests_df is None:
        log.critical("Data loading and preparation failed. Aborting.")
        sys.exit(1)

    _run_scale_scope_plots(requests_df, actual_request_id_col, MODULE_PLOT_DIR)

    log.info(
        "Scale and Scope Analysis finished. Plots saved in %s", MODULE_PLOT_DIR
    )


if __name__ == "__main__":
    setup_logging(level="INFO")
    main()
