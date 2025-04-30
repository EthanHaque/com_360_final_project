import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from warcio.archiveiterator import ArchiveIterator
from warcio.recordloader import ArcWarcRecord

from com360.logging_config import get_logger, setup_logging

log = get_logger(__name__)


def _get_encoding_from_content_type(content_type: str) -> str | None:
    """Parse the charset from a Content-Type header string.

    Parameters
    ----------
    content_type : str
        The value of the Content-Type header.

    Returns
    -------
    Optional[str]
        The detected encoding (e.g., 'utf-8'), or None if not found
        or invalid.
    """
    if not content_type or "charset=" not in content_type.lower():
        return None

    try:
        charset_part = content_type.lower().split("charset=")[-1]
        detected_encoding = charset_part.split(";")[0].strip()

        if not detected_encoding:
            return None

        _ = "test".encode(detected_encoding)
        return detected_encoding
    except LookupError:
        log.warning(
            "Detected encoding '%s' is invalid/unknown in Python.",
            detected_encoding,
        )
        return None
    except Exception as e:
        log.warning(
            "Could not parse charset from Content-Type '%s': %s",
            content_type,
            e,
        )
        return None


def _process_record(record: ArcWarcRecord) -> dict[str, Any]:
    """Process a WARC response record, extracting metadata and content.

    Parameters
    ----------
    record : warcio.ArcWarcRecord
        The WARC record to process.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing processed information:
        'url', 'status', 'content_type', 'length', 'payload' (bytes),
        'encoding' (Optional[str]), 'text' (Optional[str]),
        'error' (Optional[str]).
    """
    result: dict[str, Any] = {
        "url": record.rec_headers.get_header("WARC-Target-URI"),
        "status": record.http_headers.get_statuscode(),
        "content_type": record.http_headers.get_header(
            "Content-Type", "unknown"
        ),
        "length": int(record.rec_headers.get_header("Content-Length", 0)),
        "payload": None,
        "encoding": None,
        "text": None,
        "error": None,
    }

    try:
        result["payload"] = record.content_stream().read()

        if result["content_type"].startswith("text/"):
            detected_encoding = _get_encoding_from_content_type(
                result["content_type"]
            )
            encoding_to_use = detected_encoding or "utf-8"
            result["encoding"] = encoding_to_use

            try:
                result["text"] = result["payload"].decode(
                    encoding_to_use, errors="replace"
                )
            except Exception as decode_err:
                result["error"] = (
                    f"Decoding failed with '{encoding_to_use}': {decode_err}"
                )
                log.error(
                    "Decoding error for %s: %s", result["url"], result["error"]
                )

    except Exception as e:
        result["error"] = f"Error reading payload or processing: {e}"
        log.error(
            "Processing error for %s: %s",
            result.get("url", "N/A"),
            result["error"],
        )

    return result


def view_warc_content(
    warc_path: Path,
    max_records: int | None = 10,
    show_content: bool = True,
    max_content_length: int = 2000,
):
    """Iterate through a WARC file, processing and printing response records.

    Parameters
    ----------
    warc_path : Path
        Path to the .warc or .warc.gz file.
    max_records : Optional[int], optional
        Maximum number of records to process, by default 10.
        Set to None to process all.
    show_content : bool, optional
        Whether to print a snippet of the text content, by default True.
    max_content_length: int
        How much of the content to show if `show_content` is True. Default 2000 characters.
    """
    log.info("Processing WARC file: %s", warc_path)
    count = 0
    processed_count = 0

    try:
        with warc_path.open("rb") as stream:
            for record in ArchiveIterator(stream):
                processed_count += 1
                if record.rec_type == "response":
                    data = _process_record(record)

                    print("-" * 80)
                    print(f"URL:            {data['url']}")
                    print(f"Status:         {data['status']}")
                    print(f"Content-Type:   {data['content_type']}")
                    print(f"Length:         {data['length']} bytes")

                    if data["text"] is not None:
                        print(f"Detected/Used Encoding: {data['encoding']}")
                        if show_content:
                            snippet = data["text"][:max_content_length].strip()
                            ellipsis = (
                                "..."
                                if len(data["text"]) > max_content_length
                                else ""
                            )
                            print("\n--- Content Snippet ---")
                            print(f"{snippet}{ellipsis}")
                            print("--- End Snippet ---")
                        else:
                            print("(Text content available but not shown)")
                    elif data["payload"] is not None:
                        print("(Binary content detected)")

                    if data["error"]:
                        print(
                            f"*** Error during processing: {data['error']} ***"
                        )

                    print("-" * 80 + "\n")

                    count += 1
                    if max_records is not None and count >= max_records:
                        log.info(
                            "Reached max records to show (%d). Stopping.",
                            max_records,
                        )
                        break
                else:
                    log.debug("Skipping record type: %s", record.rec_type)

            log.info(
                "Finished processing. Showed %d response records (processed %d total records).",
                count,
                processed_count,
            )

    except FileNotFoundError:
        log.critical("Error: WARC file not found at %s", warc_path)
        sys.exit(1)
    except Exception as e:
        log.critical("An unexpected error occurred: %s", e, exc_info=True)
        sys.exit(1)


def main():
    """Command-line argument parsing and main script execution."""
    parser = argparse.ArgumentParser(
        description="Read and display content from WARC response records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "warc_file",
        type=Path,
        help="Path to the input WARC file (.warc or .warc.gz).",
    )
    parser.add_argument(
        "-n",
        "--max-records",
        type=int,
        default=10,
        help="Maximum number of response records to display content for (0 for all).",
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Do not display snippets of text content (only show metadata).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Enable verbose (DEBUG level) logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="log_level",
        const=logging.WARNING,
        help="Suppress informational (INFO level) logging.",
    )

    args = parser.parse_args()

    max_rec = args.max_records if args.max_records > 0 else None
    show_content_flag = not args.no_content

    view_warc_content(
        args.warc_file, max_records=max_rec, show_content=show_content_flag
    )


if __name__ == "__main__":
    setup_logging(level="INFO")
    main()
