# -*- coding: utf-8 -*-
"""
Converts a large CSV file containing Google Copyright Removal request data
into the more efficient Parquet format using Polars.

This script is designed to handle potentially larger-than-memory CSV files
by utilizing Polars' scanning and streaming capabilities (`scan_csv` and
`sink_parquet`).
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

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


DATA_DIR = Path("data") / "google-websearch-copyright-removals"
DEFAULT_INPUT_FILENAME = DATA_DIR / "requests.csv"
DEFAULT_OUTPUT_FILENAME = DATA_DIR / "requests.parquet"
DEFAULT_PARQUET_COMPRESSION = "zstd"


def define_requests_schema() -> dict[str, pl.DataType]:
    """
    Defines the expected schema for the requests.csv file.

    Specifying the schema upfront is crucial for performance and memory
    efficiency when scanning large files, as it avoids type inference.

    Returns
    -------
    dict[str, pl.DataType]
        A dictionary mapping column names to Polars data types.
    """
    log = get_logger(__name__)
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


def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    schema: dict[str, pl.DataType],
    compression: str = DEFAULT_PARQUET_COMPRESSION,
) -> None:
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
    schema : dict[str, pl.DataType]
        The predefined schema for the CSV file. Passed to `pl.scan_csv`.
    compression : str, optional
        The compression algorithm to use for the Parquet file,
        by default DEFAULT_PARQUET_COMPRESSION ('zstd').

    Raises
    ------
    FileNotFoundError
        If the input CSV file does not exist.
    pl.PolarsError
        If any Polars-specific error occurs during scanning or writing.
    Exception
        For any other unexpected errors.
    """
    log = get_logger(__name__)

    if not csv_path.is_file():
        log.error("Input CSV file not found.", path=str(csv_path))
        raise FileNotFoundError(f"Input CSV file not found at: {csv_path}")

    log.info(
        "Starting conversion.",
        input_path=str(csv_path),
        output_path=str(parquet_path),
        compression=compression,
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
                "File sizes",
                input_mb=round(csv_size_mb, 2),
                output_mb=round(parquet_size_mb, 2),
            )
        except Exception as stat_e:
            log.warning("Could not retrieve file sizes.", error=str(stat_e))

    except pl.PolarsError:
        log.exception("Polars error during conversion.")
        raise
    except Exception:
        log.exception("An unexpected error occurred during conversion.")
        raise


def main() -> None:
    """
    Parses command-line arguments and initiates the CSV to Parquet conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert a large Google Copyright requests CSV file to Parquet format using Polars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT_FILENAME),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILENAME),
        help="Path for the output Parquet file.",
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

    try:
        requests_schema = define_requests_schema()

        convert_csv_to_parquet(
            csv_path=args.input,
            parquet_path=args.output,
            schema=requests_schema,
            compression=args.compression,
        )
        log.info("Conversion process completed successfully.")

    except FileNotFoundError:
        log.exception("File not found.")
        sys.exit(1)
    except Exception:
        log.critical("Conversion failed. See logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
