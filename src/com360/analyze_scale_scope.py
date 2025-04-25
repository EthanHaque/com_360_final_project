"""Analyze the scale and scope of DMCA takedown requests."""

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import polars as pl

from com360.logging_config import get_logger, setup_logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "google-websearch-copyright-removals"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "copyright_analysis_plots" / "scale_scope"

REQUESTS_FILENAME = "requests.parquet"

DATE_COL = "Date"
REPORTING_ORG_COL = "Reporting organization name"
COPYRIGHT_OWNER_COL = "Copyright owner name"
REQUEST_ID_COL = "Request ID"

log = get_logger(__name__)


def load_parquet_data(
    file_path: Path, columns: list[str] | None = None
) -> pl.DataFrame | None:
    """Load specified columns from a Parquet file into a Polars DataFrame.

    Parameters
    ----------
    file_path : Path
        Specify the path to the Parquet file.
    columns : Optional[List[str]], optional
        Specify the list of columns to load. Load all if None. Defaults to None.

    Returns
    -------
    Optional[pl.DataFrame]
        Return the loaded Polars DataFrame, or None if loading fails.
    """
    log.info(
        "Attempting to load Parquet data.",
        path=str(file_path),
        columns=columns or "all",
    )
    start_time = perf_counter()
    try:
        df = pl.read_parquet(file_path, columns=columns)
        load_time = perf_counter() - start_time
        log.info(
            "Successfully loaded Parquet data.",
            path=str(file_path),
            rows=len(df),
            duration_sec=round(load_time, 2),
        )
        return df
    except FileNotFoundError:
        log.error("Parquet file not found.", path=str(file_path))
        return None
    except Exception:
        log.exception("Error loading Parquet data.", path=str(file_path))
        return None


def plot_top_n_by_request_count(
    df: pl.DataFrame,
    group_col: str,
    request_id_col: str | None = None,
    output_dir: Path = DEFAULT_PLOT_DIR,
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
    request_id_col : Optional[str], optional
        Define the column name of the unique request identifier. If None, count rows per group.
        Defaults to None.
    output_dir : Path, optional
        Set the directory to save the plot image. Defaults to DEFAULT_PLOT_DIR.
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
                f"Request ID column '{request_id_col}' not found. Counting rows instead.",
                columns=df.columns,
            )
            request_id_col = None

        if request_id_col:
            log.debug(
                "Counting distinct requests using column.",
                request_col=request_id_col,
            )
            top_items_pl = (
                df.group_by(group_col)
                .agg(pl.n_unique(request_id_col).alias("request_count"))
                .sort("request_count", descending=True)
                .head(top_n)
            )
        else:
            log.debug("Counting rows per group.")
            top_items_pl = (
                df.group_by(group_col)
                .agg(pl.count().alias("request_count"))
                .sort("request_count", descending=True)
                .head(top_n)
            )

        if top_items_pl.is_empty():
            log.warning(
                "No data found after aggregation to plot.", group_col=group_col
            )
            return

        top_items_pd = top_items_pl.to_pandas().set_index(group_col)[
            "request_count"
        ]

        if top_items_pd.empty:
            log.warning(
                "Pandas series is empty, cannot plot.", group_col=group_col
            )
            return

        plt.figure(figsize=(12, 8))
        top_items_pd.sort_values(ascending=True).plot(kind="barh")
        plot_title = f"{title_prefix} {top_n} {group_col.replace('_', ' ').title()} by Number of Requests"
        plt.title(plot_title)
        plt.xlabel("Number of Requests")
        plt.ylabel(group_col.replace("_", " ").title())
        plt.tight_layout()

        save_path = output_dir / f"{plot_filename_base}_by_requests.png"
        plt.savefig(save_path)
        log.info(f"Saved top {group_col} plot.", path=str(save_path))
        plt.close()

    except Exception:
        log.exception(f"Failed to generate top {group_col} plot.")


def plot_monthly_requests(
    requests_df: pl.DataFrame,
    date_col: str = DATE_COL,
    output_dir: Path = DEFAULT_PLOT_DIR,
) -> None:
    """Generate and save a time series plot of monthly requests.

    Parameters
    ----------
    requests_df : pl.DataFrame
        Provide the Polars DataFrame containing request data with a date column.
    date_col : str, optional
        Specify the name of the column containing datetime information. Defaults to DATE_COL.
    output_dir : Path, optional
        Set the directory to save the plot image. Defaults to DEFAULT_PLOT_DIR.
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
                pl.col(date_col).dt.truncate("1mo").alias("Month")
            )
            .agg(pl.count().alias("request_count"))
            .sort("Month")
        )

        if monthly_counts_pl.is_empty():
            log.warning("No monthly request data found to plot.")
            return

        monthly_counts_pd = monthly_counts_pl.to_pandas().set_index("Month")[
            "request_count"
        ]

        if monthly_counts_pd.empty:
            log.warning("Monthly counts Pandas series is empty, cannot plot.")
            return

        plt.figure(figsize=(14, 6))
        monthly_counts_pd.plot(kind="line", marker=".", linestyle="-")
        plt.title("Monthly Copyright Takedown Requests Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Requests")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        save_path = output_dir / "monthly_requests_timeseries.png"
        plt.savefig(save_path)
        log.info("Saved monthly requests plot.", path=str(save_path))
        plt.close()

    except Exception:
        log.exception("Failed to generate monthly requests plot.")


def main() -> None:
    """Load data and run scale/scope analyses."""
    log.info("Starting Scale and Scope Analysis.")

    DEFAULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Plot output directory.", path=str(DEFAULT_PLOT_DIR))

    requests_path = DEFAULT_DATA_DIR / REQUESTS_FILENAME

    required_cols = [
        DATE_COL,
        REPORTING_ORG_COL,
        COPYRIGHT_OWNER_COL,
    ]
    potential_request_id_col = REQUEST_ID_COL
    requests_df_full = load_parquet_data(requests_path)

    if requests_df_full is None:
        log.critical("Failed to load request data. Aborting analysis.")
        sys.exit(1)

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

    requests_df = requests_df_full.select(required_cols)

    # 1. Temporal Trends
    plot_monthly_requests(
        requests_df.select(DATE_COL)
    )  # Pass only needed column

    # 2. Top Senders (Reporting Orgs)
    sender_cols = [REPORTING_ORG_COL]
    if actual_request_id_col:
        sender_cols.append(actual_request_id_col)
    plot_top_n_by_request_count(
        df=requests_df.select(sender_cols),
        group_col=REPORTING_ORG_COL,
        request_id_col=actual_request_id_col,
        output_dir=DEFAULT_PLOT_DIR,
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
        output_dir=DEFAULT_PLOT_DIR,
        top_n=25,
        plot_filename_base="top_copyright_owners",
        title_prefix="Top",
    )

    log.info(
        "Scale and Scope Analysis finished. Plots saved in %s", DEFAULT_PLOT_DIR
    )


if __name__ == "__main__":
    setup_logging(level="INFO")
    main()
