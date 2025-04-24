# COM 360 Final Project: The Pirate Update

## Description

This project analyzes the [Google Web Search Copyright Removals](https://transparencyreport.google.com/copyright/overview) dataset. It includes steps for:

1.  **Data Preparation:** Converting the large, provided CSV files into the more efficient columnar Parquet format using Polars.
2.  **Data Analysis & Visualization:** Loading the processed Parquet data with Polars and generating several plots using Matplotlib to explore trends and distributions within the dataset.

## Features

* Efficient conversion of large CSV files (`requests.csv`, `domains.csv`, `urls-no-action-taken.csv`) to Parquet format using Polars streaming capabilities.
* ...

## Prerequisites

* **Python:** Version 3.12 or higher recommended.
* **git:** For cloning the repository.
* **wget** or a web browser: For downloading the dataset.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd com_360_final_project
    ```

2.  **Create and Activate a Virtual Environment:**
    It's highly recommended to use a virtual environment to isolate project dependencies.
    ```bash
    # Create the virtual environment (e.g., named .venv)
    python -m venv .venv

    # Activate the virtual environment:
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (Command Prompt/PowerShell):
    # .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the package and required Python libraries.
    ```bash
    pip install -e .
    ```

## Data Acquisition

1.  **Download the Data:** The dataset is provided as a ZIP archive from Google's Transparency Report. Download it using `wget` or your browser:
    ```bash
    wget [https://storage.googleapis.com/transparencyreport/google-websearch-copyright-removals.zip](https://storage.googleapis.com/transparencyreport/google-websearch-copyright-removals.zip)
    ```
    * **Note:** This is a large file (tens of gigabytes when uncompressed). Ensure you have sufficient disk space.

2.  **Unzip the Data:** Create the target directory structure and unzip the archive into it. The scripts expect the CSV files to be located in `data/google-websearch-copyright-removals/` by default.
    ```bash
    # Create directories if they don't exist (using standard commands)
    mkdir -p data/google-websearch-copyright-removals

    # Unzip (use appropriate command for your OS, e.g., 'unzip' on Linux/macOS)
    unzip google-websearch-copyright-removals.zip -d data/google-websearch-copyright-removals/
    ```

3.  **Verify:** You should now have the following CSV files in the `data/google-websearch-copyright-removals/` directory:
    * `requests.csv`
    * `domains.csv`
    * `urls-no-action-taken.csv`
    * `README.txt`

## Usage

Ensure you have **activated the virtual environment** (`source .venv/bin/activate` or `.venv\Scripts\activate`) in your terminal session before running these scripts.

1.  **Convert CSV Data to Parquet:**
    This script reads the large CSV files and converts them into the more efficient Parquet format, saving them to an output directory. This step can take a significant amount of time due to the data size.
    ```bash
    python scripts/convert_csv_to_parquet.py
    ```
    * By default, it reads from `./data/google-websearch-copyright-removals/` and writes Parquet files to `./data/google-websearch-copyright-removals/`.
    * You can specify different directories using `--input-dir` and `--output-dir`.
    * Use the `-v` or `--verbose` flag for more detailed DEBUG logging. Example:
        ```bash
        python scripts/convert_csv_to_parquet.py --input-dir path/to/csvs --output-dir path/to/parquet -v
        ```

2.  **Analyze Data and Generate Plots:**
    This script reads the generated Parquet files and creates the analysis plots.
    ```bash
    python scripts/analyze_copyright_data.py
    ```
    * By default, it reads Parquet files from `./data/google-websearch-copyright-removals/`.
    * Plots are saved to `./copyright_analysis_plots/` by default.

## Logging

* The scripts use structured logging via `structlog`.
* The default log level is `INFO`.
* You can increase verbosity by setting the `LOG_LEVEL` environment variable (e.g., `export LOG_LEVEL=DEBUG` on Linux/macOS or `set LOG_LEVEL=DEBUG` on Windows CMD or `$env:LOG_LEVEL="DEBUG"` in PowerShell) before running a script, or by using the `-v` flag where available (like in the conversion script).

