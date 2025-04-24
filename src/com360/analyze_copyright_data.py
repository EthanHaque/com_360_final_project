"""Analyzes Google Web Search Copyright Removal data using Polars and Parquet."""

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import polars as pl

try:
    from com360.logging_config import get_logger, setup_logging
except ImportError as e:
    print(
        f"Error: Could not import logging configuration from 'com360.logging_config'. "
        f"Ensure the project is installed correctly (e.g., `pip install -e .` from project root). "
        f"Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)

DEFAULT_DATA_DIR = Path("data") / "google-websearch-copyright-removals"
DEFAULT_PLOT_DIR = Path("./copyright_analysis_plots")

REQUESTS_FILENAME = "requests.parquet"
DOMAINS_FILENAME = "domains.parquet"

log = get_logger(__name__)


def load_parquet_data(
    file_path: Path, columns: list[str] | None = None
) -> pl.DataFrame | None:
    """Load specified columns from a Parquet file into a Polars DataFrame.

    Parameters
    ----------
    file_path : Path
        The path to the Parquet file.
    columns : Optional[List[str]], optional
        List of columns to load. Loads all if None. Defaults to None.

    Returns
    -------
    Optional[pl.DataFrame]
        Loaded Polars DataFrame, or None if loading fails.
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


def plot_monthly_requests(requests_df: pl.DataFrame, output_dir: Path) -> None:
    """Generate and saves a time series plot of monthly requests using Polars.

    Parameters
    ----------
    requests_df : pl.DataFrame
        Polars DataFrame containing request data with a 'Date' column (Datetime type).
    output_dir : Path
        Directory to save the plot image.
    """
    log.info("Generating monthly requests time series plot...")
    try:
        monthly_counts_pl = (
            requests_df.group_by(
                pl.col("Date").dt.truncate("1mo").alias("Month")
            )
            .agg(pl.count().alias("count"))
            .sort("Month")
        )

        monthly_counts_pd = monthly_counts_pl.to_pandas().set_index("Month")[
            "count"
        ]

        if monthly_counts_pd.empty:
            log.warning("No monthly request data found to plot.")
            return

        plt.figure(figsize=(12, 5))
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


def plot_top_domains_by_removals(
    domains_df: pl.DataFrame, output_dir: Path, top_n: int = 10
) -> None:
    """Generate and saves a bar chart of top N domains by total URLs removed using Polars.

    Parameters
    ----------
    domains_df : pl.DataFrame
        Polars DataFrame containing domain data with 'Domain' and 'URLs removed'.
    output_dir : Path
        Directory to save the plot image.
    top_n : int, optional
        Number of top domains to display. Defaults to 10.
    """
    log.info("Generating top domains plot.", top_n=top_n)
    try:
        top_domains_pl = (
            domains_df.group_by("Domain")
            .agg(pl.sum("URLs removed").alias("total_removed"))
            .sort("total_removed", descending=True)
            .head(top_n)
        )

        top_domains_pd = top_domains_pl.to_pandas().set_index("Domain")[
            "total_removed"
        ]

        if top_domains_pd.empty:
            log.warning("No domain removal data found to plot.")
            return

        plt.figure(figsize=(12, 6))
        top_domains_pd.sort_values(ascending=True).plot(kind="barh")
        plt.title(f"Top {top_n} Domains by Total URLs Removed")
        plt.xlabel("Total URLs Removed")
        plt.ylabel("Domain")
        plt.tight_layout()
        save_path = output_dir / "top_domains_by_removals.png"
        plt.savefig(save_path)
        log.info("Saved top domains plot.", path=str(save_path))
        plt.close()

    except Exception:
        log.exception("Failed to generate top domains plot.")


def plot_outcome_distribution(
    requests_df: pl.DataFrame, output_dir: Path
) -> None:
    """Generate and saves a pie chart of URL outcome distribution using Polars.

    Parameters
    ----------
    requests_df : pl.DataFrame
        Polars DataFrame containing request data with outcome columns.
    output_dir : Path
        Directory to save the plot image.
    """
    log.info("Generating outcome distribution plot...")
    outcome_cols = [
        "URLs removed",
        "URLs that were not in Google's search index",
        "URLs for which we took no action",
        "URLs pending review",
    ]

    try:
        outcomes_pl = requests_df.select(outcome_cols).sum()

        if outcomes_pl.height == 0 or outcomes_pl.width == 0:
            log.warning("No outcome data found to plot distribution.")
            return

        outcomes_pd = (
            outcomes_pl.transpose(
                include_header=True,
                header_name="outcome",
                column_names=["total"],
            )
            .to_pandas()
            .set_index("outcome")["total"]
        )

        outcomes_pd = outcomes_pd[outcomes_pd > 0]

        if outcomes_pd.empty:
            log.warning("No non-zero outcome data found to plot distribution.")
            return

        plt.figure(figsize=(8, 8))
        outcomes_pd.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white"},
        )
        plt.title("Outcome Distribution of URLs in Takedown Requests")
        plt.ylabel("")
        plt.tight_layout()
        save_path = output_dir / "outcome_distribution_pie.png"
        plt.savefig(save_path)
        log.info("Saved outcome distribution plot.", path=str(save_path))
        plt.close()

    except Exception:
        log.exception("Failed to generate outcome distribution plot.")


def plot_urls_removed_histogram(
    requests_df: pl.DataFrame, output_dir: Path
) -> None:
    """Generate and saves a histogram of URLs removed per request using Polars.

    Parameters
    ----------
    requests_df : pl.DataFrame
        Polars DataFrame containing request data with 'URLs removed'.
    output_dir : Path
        Directory to save the plot image.
    """
    log.info("Generating URLs removed per request histogram...")
    try:
        urls_removed_series = requests_df.select("URLs removed").drop_nulls()

        if urls_removed_series.height == 0:
            log.warning("No valid 'URLs removed' data found to plot histogram.")
            return

        urls_removed_data = urls_removed_series["URLs removed"].to_numpy()

        plt.figure(figsize=(10, 5))

        max_val = urls_removed_series.max().item()
        use_log_scale = max_val > 1000 if max_val is not None else False

        plt.hist(
            urls_removed_data,
            bins=50,
            edgecolor="black",
            log=use_log_scale,
        )
        plt.title(
            "Distribution of URLs Removed per Request"
            + (" (Log Scale)" if use_log_scale else "")
        )
        plt.xlabel("Number of URLs Removed per Request")
        plt.ylabel(
            "Number of Requests" + (" (Log Scale)" if use_log_scale else "")
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        save_path = output_dir / "urls_removed_histogram.png"
        plt.savefig(save_path)
        log.info("Saved URLs removed histogram.", path=str(save_path))
        plt.close()

    except Exception:
        log.exception("Failed to generate URLs removed histogram.")


def main() -> None:
    """Load data using Polars and generate all plots."""
    setup_logging(level="INFO")
    log.info("Analysis script starting.")

    DEFAULT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Plot output directory.", path=str(DEFAULT_PLOT_DIR))

    requests_path = DEFAULT_DATA_DIR / REQUESTS_FILENAME
    domains_path = DEFAULT_DATA_DIR / DOMAINS_FILENAME

    requests_cols_needed = [
        "Date",
        "URLs removed",
        "URLs that were not in Google's search index",
        "URLs for which we took no action",
        "URLs pending review",
    ]
    requests_df = load_parquet_data(requests_path, columns=requests_cols_needed)

    domains_cols_needed = ["Domain", "URLs removed"]
    domains_df = load_parquet_data(domains_path, columns=domains_cols_needed)

    if requests_df is not None:
        plot_monthly_requests(requests_df.select("Date"), DEFAULT_PLOT_DIR)
        plot_outcome_distribution(
            requests_df.select(
                [
                    "URLs removed",
                    "URLs that were not in Google's search index",
                    "URLs for which we took no action",
                    "URLs pending review",
                ]
            ),
            DEFAULT_PLOT_DIR,
        )
        plot_urls_removed_histogram(
            requests_df.select("URLs removed"), DEFAULT_PLOT_DIR
        )
    else:
        log.warning(
            "Skipping plots requiring requests data due to loading error."
        )

    if domains_df is not None:
        plot_top_domains_by_removals(domains_df, DEFAULT_PLOT_DIR)
    else:
        log.warning(
            "Skipping plots requiring domains data due to loading error."
        )

    log.info("Analysis script finished. Plots saved in %s", DEFAULT_PLOT_DIR)


if __name__ == "__main__":
    main()
