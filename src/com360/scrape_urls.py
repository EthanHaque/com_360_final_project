"""Scrape URLs from dataset using asynchronous requests."""

import argparse
import asyncio
import io
import signal
import sys
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from time import perf_counter

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm_asyncio
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

from com360.logging_config import get_logger, setup_logging

log = get_logger(__name__)
warc_writer_lock = asyncio.Lock()
resume_file_lock = asyncio.Lock()
shutdown_event = asyncio.Event()

DEFAULT_DATA_DIR = Path("data") / "google-websearch-copyright-removals"
DEFAULT_INPUT_FILENAME = "requests.parquet"
DEFAULT_URL_COLUMN = "Lumen URL"
DEFAULT_OUTPUT_DIR = Path("data") / "scraped_warc"
DEFAULT_RESUME_DIR = Path("data") / "scraper_state"
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY_SEC = 2
DEFAULT_BATCH_SIZE = 100
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; COM360DataScraper/0.1)"


def human_bytes(num_bytes: float) -> str:
    """Convert bytes to a human-readable format (KiB, MiB, GiB).

    Parameters
    ----------
    num_bytes : float
        Number of bytes.

    Returns
    -------
    str
        Human-readable string representation of bytes.
    """
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TiB"


async def load_completed_urls(resume_file: Path) -> set[str]:
    """Load previously completed URLs from the resume file.

    Parameters
    ----------
    resume_file : Path
        Path to the file storing completed URLs (one per line).

    Returns
    -------
    Set[str]
        A set of URLs that have already been successfully scraped.
    """
    completed: set[str] = set()
    if not resume_file.is_file():
        log.info("Resume file not found, starting fresh.", path=resume_file)
        return completed

    log.info("Loading completed URLs from resume file...", path=resume_file)
    try:
        async with resume_file_lock:

            def read_file():
                urls = set()
                with resume_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped:
                            urls.add(stripped)
                return urls

            completed = await asyncio.to_thread(read_file)

    except Exception:
        log.exception(
            "Failed to load resume file, starting fresh.", path=resume_file
        )
        return set()

    log.info(f"Loaded {len(completed):,} completed URLs.")
    return completed


async def append_completed_url(resume_file: Path, url: str):
    """Append a successfully scraped URL to the resume file.

    Parameters
    ----------
    resume_file : Path
        Path to the file storing completed URLs.
    url : str
        The URL that was successfully scraped.
    """
    try:
        async with resume_file_lock:

            def append_to_file():
                with resume_file.open("a", encoding="utf-8") as f:
                    f.write(f"{url}\n")

            await asyncio.to_thread(append_to_file)
    except Exception as e:
        log.warning(
            "Failed to append URL to resume file.", url=url, error=str(e)
        )


async def fetch_url(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int,
    retries: int,
    retry_delay: float,
    user_agent: str,
) -> tuple[int | None, StatusAndHeaders | None, bytes | None]:
    """Fetch a single URL with timeout, retries, and custom User-Agent.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The shared asynchronous HTTP session.
    url : str
        The URL to fetch.
    timeout : int
        Request timeout in seconds.
    retries : int
        Number of times to retry on transient errors.
    retry_delay : float
        Seconds to wait between retries (exponential backoff applied).
    user_agent : str
        The User-Agent string to send with the request.

    Returns
    -------
    Tuple[Optional[int], Optional[StatusAndHeaders], Optional[bytes]]
        A tuple containing:
        - HTTP status code (or None on failure after retries).
        - warcio StatusAndHeaders object (or None).
        - Response body bytes (or None).
    """
    headers = {"User-Agent": user_agent}

    for attempt in range(retries + 1):
        if shutdown_event.is_set():
            log.warning("Shutdown requested, aborting fetch.", url=url)
            return None, None, None
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers=headers,
                allow_redirects=True,
                max_redirects=10,
            ) as response:
                content = await response.read()

                protocol = (
                    f"HTTP/{response.version.major}.{response.version.minor}"
                )
                header_list = [
                    (k.decode("latin-1"), v.decode("latin-1"))
                    for k, v in response.raw_headers
                ]
                warc_headers = StatusAndHeaders(
                    f"{response.status} {response.reason}",
                    header_list,
                    protocol=protocol,
                )

                log.debug(
                    "Fetch successful.",
                    url=url,
                    status=response.status,
                    size=human_bytes(len(content)),
                )
                return response.status, warc_headers, content

        except (
            TimeoutError,
            aiohttp.ClientError,
            aiohttp.http_exceptions.HttpProcessingError,
        ) as e:
            log.warning(
                "Fetch attempt failed.",
                url=url,
                attempt=attempt + 1,
                max_attempts=retries + 1,
                error_type=type(e).__name__,
                error=str(e),
            )
            if attempt < retries:
                delay = retry_delay * (2**attempt)
                log.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
            else:
                log.error(
                    "Fetch failed after all retries.",
                    url=url,
                    error_type=type(e).__name__,
                    error=str(e),
                )

        except Exception:
            log.exception(
                "Unexpected error during fetch.", url=url, attempt=attempt + 1
            )
            break

    return None, None, None


async def write_warc_record(
    writer: WARCWriter,
    url: str,
    status_code: int,
    warc_headers: StatusAndHeaders,
    payload: bytes,
):
    """Write a fetched response to the WARC file using warcio.

    Uses an asyncio Lock to ensure writes are serialized, as warcio writer
    might not be inherently async-safe for concurrent writes from tasks.

    Parameters
    ----------
    writer : WARCWriter
        The warcio writer instance.
    url : str
        The original request URL.
    status_code : int
        The HTTP status code received.
    warc_headers : StatusAndHeaders
        The warcio StatusAndHeaders object representing response headers.
    payload : bytes
        The response body content.
    """
    try:
        payload_stream = io.BytesIO(payload)
        response_record = writer.create_warc_record(
            url,
            "response",
            payload=payload_stream,
            http_headers=warc_headers,
        )
        async with warc_writer_lock:
            await asyncio.to_thread(writer.write_record, response_record)

        log.debug("Wrote WARC record.", url=url, status=status_code)

    except Exception:
        log.exception("Failed to write WARC record.", url=url)


async def scrape_worker(
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    writer: WARCWriter,
    progress_bar: tqdm_asyncio,
    resume_file: Path,
    timeout: int,
    retries: int,
    retry_delay: float,
    user_agent: str,
):
    """Consumes URLs from the queue, fetches, and writes to WARC.

    Parameters
    ----------
    queue : asyncio.Queue
        Queue from which to get URLs.
    session : aiohttp.ClientSession
        Shared HTTP session.
    writer : WARCWriter
        Shared WARC writer instance.
    progress_bar : tqdm_asyncio
        Progress bar instance to update.
    resume_file : Path
        Path to the resume file for logging completion.
    timeout : int
        Request timeout in seconds.
    retries : int
        Number of fetch retries.
    retry_delay : float
        Base delay between retries.
    user_agent : str
        User-Agent string for requests.
    """
    processed_count = 0
    success_count = 0
    fail_count = 0

    while not shutdown_event.is_set():
        try:
            url = await queue.get()
            if url is None:
                queue.put_nowait(None)
                break

            log.debug("Processing URL from queue.", url=url)
            status, headers, content = await fetch_url(
                session, url, timeout, retries, retry_delay, user_agent
            )

            if (
                status is not None
                and headers is not None
                and content is not None
            ):
                await write_warc_record(writer, url, status, headers, content)
                await append_completed_url(resume_file, url)
                success_count += 1
            else:
                log.warning(
                    "Skipping WARC record due to fetch failure.", url=url
                )
                fail_count += 1

            progress_bar.update(1)
            processed_count += 1
            queue.task_done()

        except asyncio.CancelledError:
            log.info("Worker cancelled.")
            break
        except Exception:
            log.exception("Error in scrape worker.")
            with suppress(ValueError):
                queue.task_done()

    log.info(
        "Worker finished.",
        processed=processed_count,
        successful=success_count,
        failed=fail_count,
    )


async def url_producer(
    queue: asyncio.Queue,
    parquet_path: Path,
    url_column: str,
    completed_urls: set[str],
    batch_size: int,
):
    """Read URLs from Parquet file and puts them onto the queue.

    Skips URLs that are already in the completed_urls set.

    Parameters
    ----------
    queue : asyncio.Queue
        Queue to put URLs onto.
    parquet_path : Path
        Path to the input Parquet file.
    url_column : str
        Name of the column containing URLs in the Parquet file.
    completed_urls : Set[str]
        Set of URLs already scraped (from resume file).
    batch_size : int
        Number of rows to read from Parquet at a time (for memory efficiency).
    """
    log.info("Starting URL producer...", path=str(parquet_path))
    total_queued = 0
    total_skipped = 0
    total_read = 0

    try:
        scanner = pl.scan_parquet(parquet_path)
        scanner = scanner.select(pl.col(url_column))

        for batch_df in scanner.collect(streaming=True).iter_slices(
            n_rows=batch_size
        ):
            if shutdown_event.is_set():
                log.warning("Shutdown requested, stopping URL producer.")
                break

            urls_in_batch = batch_df[url_column].to_list()
            total_read += len(urls_in_batch)

            for url in urls_in_batch:
                if (
                    not url
                    or not isinstance(url, str)
                    or not url.startswith(("http://", "https://"))
                ):
                    log.warning("Skipping invalid URL.", url=repr(url))
                    total_skipped += 1
                    continue

                if url in completed_urls:
                    log.debug("Skipping already completed URL.", url=url)
                    total_skipped += 1
                    continue

                await queue.put(url)
                total_queued += 1
                # Avoid queue getting excessively large if workers are slow
                if queue.qsize() > batch_size * 5:  # Heuristic threshold
                    log.debug("Queue size large, pausing producer briefly...")
                    await asyncio.sleep(1)

                if shutdown_event.is_set():
                    log.warning("Shutdown requested, stopping URL producer.")
                    break

            log.debug(
                f"Producer batch complete. Queued: {total_queued}, Skipped: {total_skipped}, Read: {total_read}"
            )

        log.info(
            "URL Producer finished.",
            total_urls_read=total_read,
            total_urls_queued=total_queued,
            total_urls_skipped=total_skipped,
        )

    except pl.exceptions.ColumnNotFoundError:
        log.critical(
            f"URL column '{url_column}' not found in Parquet file.",
            path=str(parquet_path),
        )
        shutdown_event.set()
    except FileNotFoundError:
        log.critical("Input Parquet file not found.", path=str(parquet_path))
        shutdown_event.set()
    except Exception:
        log.exception("Error during URL production.")
        shutdown_event.set()
    finally:
        log.info("Producer signaling end of queue...")


def build_signal_handler(queue: asyncio.Queue, n_workers: int):
    """Return a signal handler that knows the queue + worker count."""

    def handle_signal(signum, frame):
        log.warning("Signal %s received – shutting down…", signum)
        shutdown_event.set()
        loop = asyncio.get_running_loop()
        for _ in range(n_workers):
            loop.call_soon_threadsafe(queue.put_nowait, None)

    return handle_signal


async def main_async(
    queue: asyncio.Queue[str | None], args: argparse.Namespace
):
    """Asynchronous function to set up and run the scraper."""
    start_time = perf_counter()
    log.info("Starting asynchronous web scraper.", args=vars(args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.resume_dir.mkdir(parents=True, exist_ok=True)
    warc_path = (
        args.output_dir / f"scraped_{Path(args.input_path).stem}.warc.gz"
    )
    resume_file = (
        args.resume_dir / f"{Path(args.input_path).stem}_completed.txt"
    )
    log.info(f"Output WARC file: {warc_path}")
    log.info(f"Resume state file: {resume_file}")

    completed_urls = await load_completed_urls(resume_file)
    initial_completed_count = len(completed_urls)

    async with AsyncExitStack() as stack:
        connector = aiohttp.TCPConnector(
            limit_per_host=args.concurrency // 5,
            enable_cleanup_closed=True,
        )
        session = await stack.enter_async_context(
            aiohttp.ClientSession(connector=connector, raise_for_status=False)
        )
        log.info("aiohttp session created.")

        warc_file = await asyncio.to_thread(lambda: Path.open(warc_path, "wb"))
        stack.push_async_callback(lambda: asyncio.to_thread(warc_file.close))
        writer = WARCWriter(warc_file, gzip=True)
        log.info("WARC writer initialized.")

        estimated_total = 0
        try:
            count_df = (
                pl.scan_parquet(args.input_path).select(pl.len()).collect()
            )
            estimated_total = count_df[0, 0] - initial_completed_count
            if estimated_total < 0:
                estimated_total = 0
            log.info(
                f"Estimated URLs to scrape: {estimated_total:,} (Total: {count_df[0, 0]:,}, Completed: {initial_completed_count:,})"
            )
        except Exception as e:
            log.warning(f"Could not estimate total URLs: {e}")

        progress_bar = tqdm_asyncio(
            total=estimated_total if estimated_total > 0 else None,
            unit=" URL",
            desc="Scraping",
        )

        producer_task = asyncio.create_task(
            url_producer(
                queue,
                args.input_path,
                args.url_column,
                completed_urls,
                args.batch_size,
            )
        )

        worker_tasks = []
        for _ in range(args.concurrency):
            task = asyncio.create_task(
                scrape_worker(
                    queue,
                    session,
                    writer,
                    progress_bar,
                    resume_file,
                    args.timeout,
                    args.retries,
                    args.retry_delay,
                    args.user_agent,
                )
            )
            worker_tasks.append(task)

        await producer_task

        if not shutdown_event.is_set():
            log.info("Producer finished, adding worker stop signals...")

            for _ in range(args.concurrency):
                await queue.put(None)

            log.info("Waiting for workers to process remaining queue items...")

            await queue.join()

        log.info("All queue items processed. Shutting down workers...")
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        log.info("Scraping tasks complete.")

    end_time = perf_counter()
    duration = end_time - start_time
    final_completed_urls = await load_completed_urls(resume_file)
    newly_completed = len(final_completed_urls) - initial_completed_count

    log.info(f"Total duration: {duration:.2f} seconds")
    log.info(f"Input Parquet: {args.input_path}")
    log.info(
        f"Output WARC: {warc_path} ({human_bytes(warc_path.stat().st_size if warc_path.exists() else 0)})"
    )
    log.info(f"Resume file: {resume_file}")
    log.info(f"Initially completed URLs: {initial_completed_count:,}")
    log.info(f"Newly completed URLs: {newly_completed:,}")
    log.info(f"Total completed URLs now: {len(final_completed_urls):,}")

    if shutdown_event.is_set():
        log.warning("Process was interrupted by a signal.")
    log.info("Scraper finished.")


def main():
    """Parse command-line arguments and runs the scraper."""
    parser = argparse.ArgumentParser(
        description="Efficiently scrape URLs from a Parquet file and save to WARC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input Parquet file containing URLs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output WARC file(s).",
    )
    parser.add_argument(
        "--url-column",
        type=str,
        default=DEFAULT_URL_COLUMN,
        help="Name of the column containing URLs in the Parquet file.",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=DEFAULT_RESUME_DIR,
        help="Directory to store the file tracking completed URLs for resuming.",
    )

    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of concurrent download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="Number of retries for failed requests.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY_SEC,
        help="Initial delay (seconds) between retries (uses exponential backoff).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of URLs to read from Parquet/process in batches.",
    )

    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="User-Agent string to use for requests.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        dest="log_level",
        const="DEBUG",
        default="INFO",
        help="Enable verbose (DEBUG level) logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="log_level",
        const="WARNING",
        help="Suppress informational (INFO level) logging.",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    signal.signal(signal.SIGINT, build_signal_handler(queue, args.concurrency))
    signal.signal(signal.SIGTERM, build_signal_handler(queue, args.concurrency))

    try:
        asyncio.run(main_async(queue, args))
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt caught in main.")
    except Exception:
        log.critical("Unhandled exception in main execution.", exc_info=True)
        sys.exit(1)
    finally:
        log.info("Shutting down.")
        if shutdown_event.is_set():
            sys.exit(1)


if __name__ == "__main__":
    main()
