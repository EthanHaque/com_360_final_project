"""
Analyze DMCA takedown targeting strategies and their evolution over time
using Polars Lazy API for memory efficiency.
"""

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import polars as pl
import pandas as pd

from com360.logging_config import get_logger, setup_logging
from com360.utils import (
    apply_plot_styling,
    save_plot,
    DEFAULT_DATA_DIR,
    DEFAULT_BASE_PLOT_DIR,
)

MODULE_PLOT_DIR = DEFAULT_BASE_PLOT_DIR / "targeting_lazy"
REQUESTS_FILENAME = "requests.parquet"
DOMAINS_FILENAME = "domains.parquet"

DATE_COL = "Date"
REQUEST_ID_COL = "Request ID"
DOMAIN_COL = "Domain"
PLATFORM_CATEGORY_COL = "Platform Category"
MONTH_COL = "Month"
REQUEST_COUNT_COL = "request_count"
PROPORTION_COL = "proportion"

log = get_logger(__name__)


def categorize_domain(domain: str | None) -> str:
    """Categorize a domain name into a platform type based on simple heuristics.

    Note: This function provides a basic, example categorization. For robust analysis,
    replace this with a more sophisticated approach using extensive lists of known
    sites, refined keywords, or external classification data.

    Parameters
    ----------
    domain : str | None
        Provide the domain name string or None.

    Returns
    -------
    str
        Return the inferred platform category string.
    """
    if domain is None:
        return "Unknown"
    domain_lower = domain.lower()

    if "torrent" in domain_lower or domain_lower in {
        "thepiratebay.org",
        "1337x.to",
        "yts.mx",
        "rarbg.to",
        "rutracker.org",
    }:
        return "P2P/Torrent"
    if (
        "mega.nz" in domain_lower
        or "rapidgator" in domain_lower
        or "1fichier" in domain_lower
        or "zippyshare" in domain_lower
        or "uploaded.net" in domain_lower
    ):
        return "Cyberlocker/Hosting"
    if (
        "youtube.com" in domain_lower
        or "dailymotion.com" in domain_lower
        or "vimeo.com" in domain_lower
        or "soundcloud.com" in domain_lower
    ):
        return "UGC Platform"
    if (
        "facebook.com" in domain_lower
        or "twitter.com" in domain_lower
        or "vk.com" in domain_lower
        or "instagram.com" in domain_lower
    ):
        return "Social Media"
    if "google.com" in domain_lower or "blogspot.com" in domain_lower:
        if (
            "drive.google.com" in domain_lower
            or "docs.google.com" in domain_lower
        ):
            return "Cloud Storage Abuse"
        return "Google Service (Other)"
    if "dropbox.com" in domain_lower:
        return "Cloud Storage Abuse"
    if (
        "stream" in domain_lower
        or "movie" in domain_lower
        or "watch" in domain_lower
    ):
        return "Streaming (Generic)"

    return "Other/Unknown"


def _create_lazy_processing_plan(
    requests_path: Path, domains_path: Path
) -> pl.LazyFrame | None:
    """Create the full Polars LazyFrame processing plan.

    Scans data, joins, categorizes, aggregates, and calculates proportions lazily.

    Parameters
    ----------
    requests_path : Path
        Specify the path to the requests Parquet file.
    domains_path : Path
        Specify the path to the domains Parquet file.

    Returns
    -------
    pl.LazyFrame | None
        Return the resulting LazyFrame plan or None if setup fails.
    """
    log.info("Creating lazy processing plan...")
    try:
        requests_lf = pl.scan_parquet(requests_path).select(
            REQUEST_ID_COL, DATE_COL
        )
        domains_lf = pl.scan_parquet(domains_path).select(
            REQUEST_ID_COL, DOMAIN_COL
        )

        req_schema = pl.read_parquet_schema(requests_path)
        dom_schema = pl.read_parquet_schema(domains_path)

        if req_schema.get(REQUEST_ID_COL) != dom_schema.get(REQUEST_ID_COL):
            log.warning(
                "Request ID types differ, attempting cast in lazy plan.",
                req_type=req_schema.get(REQUEST_ID_COL),
                dom_type=dom_schema.get(REQUEST_ID_COL),
            )
            target_type = req_schema.get(REQUEST_ID_COL, pl.Utf8)
            domains_lf = domains_lf.with_columns(
                pl.col(REQUEST_ID_COL).cast(target_type)
            )

        combined_lf = domains_lf.join(
            requests_lf, on=REQUEST_ID_COL, how="inner"
        )

        categorized_lf = combined_lf.with_columns(
            pl.col(DOMAIN_COL)
            .map_elements(
                categorize_domain, return_dtype=pl.Utf8, skip_nulls=False
            )
            .alias(PLATFORM_CATEGORY_COL)
        )

        monthly_agg_lf = categorized_lf.group_by(
            pl.col(DATE_COL).dt.truncate("1mo").alias(MONTH_COL),
            PLATFORM_CATEGORY_COL,
        ).agg(pl.n_unique(REQUEST_ID_COL).alias(REQUEST_COUNT_COL))

        monthly_proportions_lf = monthly_agg_lf.with_columns(
            (
                pl.col(REQUEST_COUNT_COL)
                / pl.sum(REQUEST_COUNT_COL).over(MONTH_COL)
            ).alias(PROPORTION_COL)
        )

        final_lf = monthly_proportions_lf.sort(MONTH_COL, PLATFORM_CATEGORY_COL)

        log.info("Lazy processing plan created successfully.")
        return final_lf

    except FileNotFoundError:
        log.error(
            "Input Parquet file not found during scan setup.",
            requests_path=requests_path,
            domains_path=domains_path,
        )
        return None
    except Exception:
        log.exception("Failed to create lazy processing plan.")
        return None


def plot_platform_trends_from_long(
    long_data_df: pl.DataFrame,
    output_dir: Path = MODULE_PLOT_DIR,
) -> None:
    """Generate and save a stacked area chart from long-format aggregated data.

    Parameters
    ----------
    long_data_df : pl.DataFrame
        Provide a Polars DataFrame in long format with columns:
        'Month', 'Platform Category', 'proportion'.
    output_dir : Path, optional
        Set the directory to save the plot image. Defaults to MODULE_PLOT_DIR.
    """
    log.info("Generating platform targeting trends plot from long data...")
    try:
        if long_data_df.is_empty():
            log.warning("No aggregated trend data available to plot.")
            return

        log.debug("Pivoting aggregated data for plotting...")
        trends_pd = (
            long_data_df.pivot(
                index=MONTH_COL,
                columns=PLATFORM_CATEGORY_COL,
                values=PROPORTION_COL,
            )
            .fill_null(0.0)
            .sort(MONTH_COL)
            .to_pandas()
            .set_index(MONTH_COL)
        )
        log.debug("Pandas pivot successful.", shape=trends_pd.shape)

        max_categories_to_plot = 10
        if len(trends_pd.columns) > max_categories_to_plot:
            log.info(
                "Too many categories, aggregating less frequent ones into 'Other'."
            )
            category_sums = trends_pd.sum().sort_values(ascending=False)
            top_categories = category_sums.head(
                max_categories_to_plot - 1
            ).index.tolist()
            other_categories = category_sums.iloc[
                max_categories_to_plot - 1 :
            ].index.tolist()

            trends_plot_pd = trends_pd[top_categories].copy()
            trends_plot_pd["Other Aggregated"] = trends_pd[
                other_categories
            ].sum(axis=1)
            plot_data = trends_plot_pd
        else:
            plot_data = trends_pd

        if plot_data.empty:
            log.warning("Plotting data is empty after processing.")
            return

        fig, ax = plt.subplots()
        plot_data.plot(kind="area", stacked=True, alpha=0.8, ax=ax)

        ax.set_title(
            "Proportion of Takedown Requests by Targeted Platform Type Over Time"
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Proportion of Monthly Requests")
        ax.set_ylim(0, 1)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            title="Platform Category",
            fontsize="x-small",
        )

        save_path = output_dir / "platform_targeting_trends_lazy.png"
        save_plot(fig, save_path)
        plt.close(fig)

    except Exception:
        log.exception("Failed to generate platform targeting trends plot.")


def main() -> None:
    """Create and execute lazy plan, then plot the results."""
    log.info("Starting Targeting Strategy Analysis (Lazy API).")

    apply_plot_styling()
    MODULE_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Plot output directory.", path=str(MODULE_PLOT_DIR))

    current_working_directory = Path.cwd()
    requests_path = (
        current_working_directory / DEFAULT_DATA_DIR / REQUESTS_FILENAME
    )
    domains_path = (
        current_working_directory / DEFAULT_DATA_DIR / DOMAINS_FILENAME
    )

    # Step 1: Create the LazyFrame plan
    final_lf = _create_lazy_processing_plan(requests_path, domains_path)
    if final_lf is None:
        log.critical("Failed to create lazy processing plan. Aborting.")
        sys.exit(1)

    # Step 2: Execute the plan and collect the aggregated results
    log.info("Executing lazy plan and collecting aggregated results...")
    try:
        aggregated_results_df = final_lf.collect(streaming=False)
        log.info(
            "Lazy plan execution complete.", shape=aggregated_results_df.shape
        )
    except Exception:
        log.exception("Failed to execute lazy plan or collect results.")
        sys.exit(1)

    plot_platform_trends_from_long(
        aggregated_results_df, output_dir=MODULE_PLOT_DIR
    )

    log.info(
        "Targeting Strategy Analysis finished. Plots saved in %s",
        MODULE_PLOT_DIR,
    )


if __name__ == "__main__":
    setup_logging(level="INFO")
    main()
