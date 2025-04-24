# -*- coding: utf-8 -*-
"""
Converts large Google Copyright Removal CSV files (requests, domains,
urls-no-action-taken) into the more efficient Parquet format using Polars.

This script is designed to handle potentially larger-than-memory CSV files
by utilizing Polars' scanning and streaming capabilities (`scan_csv` and
`sink_parquet`).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict

import polars as pl

try:
    from com360.logging_config import setup_logging, get_logger
except ImportError as e:
    print(
        f"Error: Could not import logging configuration from 'com360.logging_config'. "
        f"Ensure the project is installed correctly (e.g., `pip install -e .` from project root). "
        f"Details: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_BASE_DATA_DIR = Path("data") / "google-websearch-copyright-removals"
DEFAULT_INPUT_DIR = DEFAULT_BASE_DATA_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DATA_DIR
DEFAULT_PARQUET_COMPRESSION = "zstd"

SchemaDict = Dict[str, pl.DataType]


def get_requests_schema() -> SchemaDict:
    """
    Defines the expected schema for the requests.csv file.

    Specifying the schema upfront is crucial for performance and memory
    efficiency when scanning large files, as it avoids type inference.

    Returns
    -------
    SchemaDict
        A dictionary mapping column names to Polars data types.
    """
    log = get_logger(__name__ + ".get_requests_schema")
    schema = {
        "Request ID": pl.UInt64,
        "Date": pl.Utf8,
        "Lumen URL": pl.Utf8,
        "Copyright owner ID": pl.UInt32,
        "Copyright owner name": pl.Utf8,
        "Reporting organization ID": pl.UInt32,
        "Reporting organization name": pl.Utf8,
        "URLs removed": pl.UInt32,
        "URLs that were not in Google's search index": pl.UInt32,
        "URLs for which we took no action": pl.UInt32,
        "URLs pending review": pl.UInt32,
        "From Abuser": pl.Boolean,
    }
    log.debug("Defined schema.", schema_keys=list(schema.keys()))
    return schema


def get_domains_schema() -> SchemaDict:
    """
    Defines the expected schema for the domains.csv file.

    Returns
    -------
    SchemaDict
        A dictionary mapping column names to Polars data types.
    """
    log = get_logger(__name__ + ".get_domains_schema")
    schema = {
        "Request ID": pl.UInt64,
        "Domain": pl.Utf8,
        "URLs removed": pl.UInt32,
        "URLs that were not in Google's search index": pl.UInt32,
        "URLs for which we took no action": pl.UInt32,
        "URLs pending review": pl.UInt32,
        "From Abuser": pl.Boolean,
    }
    log.debug("Defined schema.", schema_keys=list(schema.keys()))
    return schema


def get_urls_no_action_schema() -> SchemaDict:
    """
    Defines the schema for the urls-no-action-taken.csv file.

    Returns
    -------
    SchemaDict
        A dictionary mapping column names to Polars data types.
    """
    log = get_logger(__name__ + ".get_urls_no_action_schema")
    schema = {
        "Request ID": pl.UInt64,
        "Domain": pl.Utf8,
        "URL": pl.Utf8,
        "From Abuser": pl.Boolean,
    }
    log.debug("Defined schema.", schema_keys=list(schema.keys()))
    return schema


def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    schema: SchemaDict,
    parse_date_col: bool = False,
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> bool:
    """
    Converts a CSV file to Parquet format using Polars streaming.

    Reads the CSV lazily using `scan_csv` and writes directly to Parquet
    using `sink_parquet` to handle files potentially larger than memory.

    Parameters
    ----------
    csv_path : Path
        Path to the input CSV file.
    parquet_path : Path
        Path where the output Parquet file will be saved.
    schema : SchemaDict
        The predefined schema for the CSV file. Passed to `pl.scan_csv`.
    parse_date_col : bool, optional
        If True and a 'Date' column exists in the schema, attempts to parse
        it from Utf8 string to Datetime. Defaults to False.
    compression : str, optional
        The compression algorithm for the Parquet file.
        Defaults to DEFAULT_PARQUET_COMPRESSION ('zstd').

    Returns
    -------
    bool
        True if conversion was successful, False otherwise.

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    """
    log = get_logger(__name__ + ".convert_csv_to_parquet")

    if not csv_path.is_file():
        log.error("Input CSV file not found.", path=str(csv_path))
        raise FileNotFoundError(f"Input CSV file not found at: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting conversion.",
        input_path=str(csv_path),
        output_path=str(parquet_path),
        compression=compression,
        parse_date=parse_date_col,
    )
    start_time = perf_counter()

    try:
        lazy_df = pl.scan_csv(
            csv_path,
            schema=schema,
            infer_schema_length=0,
            try_parse_dates=False,
            low_memory=True,
        )

        if parse_date_col and "Date" in schema:
            log.debug("Applying date parsing transformation.")
            lazy_df = lazy_df.with_columns(
                pl.col("Date")
                .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=True)
                .alias("Date")
            )

        log.info("Writing to Parquet via sink_parquet (streaming)...")
        lazy_df.sink_parquet(parquet_path, compression=compression)

        end_time = perf_counter()
        duration = end_time - start_time
        log.info(
            "Successfully converted to Parquet.",
            duration_sec=round(duration, 2),
            output_path=str(parquet_path),
        )

        try:
            csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
            parquet_size_mb = parquet_path.stat().st_size / (1024 * 1024)
            log.info(
                "File sizes (MB)",
                input=round(csv_size_mb, 2),
                output=round(parquet_size_mb, 2),
            )
        except Exception as stat_e:
            log.warning("Could not retrieve file sizes.", error=str(stat_e))

        return True

    except pl.PolarsError as e:
        log.error(
            "Polars error during conversion.",
            error_type=type(e).__name__,
            details=str(e),
            input_path=str(csv_path),
        )
    except Exception:
        log.exception(
            "An unexpected error occurred during conversion.", input_path=str(csv_path)
        )

    return False


def main() -> None:
    """
    Parses command-line arguments and initiates the conversion process
    for predefined CSV files.
    """
    parser = argparse.ArgumentParser(
        description="Convert large Google Copyright CSV files to Parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where output Parquet files will be saved.",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=str,
        default=DEFAULT_PARQUET_COMPRESSION,
        choices=["snappy", "gzip", "zstd", "lz4", "uncompressed"],
        help="Compression algorithm for the Parquet file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    log = get_logger(__name__)

    log.debug("Arguments parsed.", args=vars(args))

    conversion_tasks = [
        {
            "input_name": "requests.csv",
            "output_name": "requests.parquet",
            "schema_func": get_requests_schema,
            "parse_date": True,
        },
        {
            "input_name": "domains.csv",
            "output_name": "domains.parquet",
            "schema_func": get_domains_schema,
            "parse_date": False,
        },
        {
            "input_name": "urls-no-action-taken.csv",
            "output_name": "urls_no_action_taken.parquet",
            "schema_func": get_urls_no_action_schema,
            "parse_date": False,
        },
    ]

    successful_conversions = 0
    failed_conversions = 0

    log.info(
        "Starting conversion for %d file(s)...",
        len(conversion_tasks),
        input_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
    )

    for task in conversion_tasks:
        input_file = args.input_dir / task["input_name"]
        output_file = args.output_dir / task["output_name"]
        schema_func = task["schema_func"]
        parse_date = task.get("parse_date", False)

        log.info("--- Processing task: %s ---", task["input_name"])

        try:
            schema = schema_func()
            success = convert_csv_to_parquet(
                csv_path=input_file,
                parquet_path=output_file,
                schema=schema,
                parse_date_col=parse_date,
                compression=args.compression,
            )
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
                log.warning(
                    "Conversion task marked as failed.", task=task["input_name"]
                )
        except FileNotFoundError:
            log.error(
                "Skipping task due to FileNotFoundError.", task=task["input_name"]
            )
            failed_conversions += 1
        except Exception:
            log.critical(
                "Skipping task due to critical error.", task=task["input_name"]
            )
            failed_conversions += 1
        finally:
            log.info("--- Finished task: %s ---", task["input_name"])

    log.info("=" * 30 + " Conversion Summary " + "=" * 30)
    log.info("Total tasks: %d", len(conversion_tasks))
    log.info("Successful conversions: %d", successful_conversions)
    log.info("Failed conversions: %d", failed_conversions)
    log.info("=" * 78)

    if failed_conversions > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
