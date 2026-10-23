"""
make_dataset.py
----------------
Data ingestion pipeline for CDC BRFSS 2024 survey dataset.

Streams the raw SAS transport format (XPT) file directly from the compressed
archive (`data/raw/LLCP2024XPT.zip`) using pyreadstat with 'latin1' encoding,
avoiding uncompressed extraction of the ~1.02 GB file.

Selects an initial set of 61 verified candidate variables covering:
  - Complex survey design and weighting (_LLCPWT, _STSTR, _PSU, _STATE)
  - Demographics & socioeconomic status
  - General health and functional status
  - Healthcare access and utilization
  - Chronic health conditions & comorbidity indicators
  - Behavioral risk factors (smoking, alcohol, exercise, BMI)
  - Disability / functional limitation indicators

Outputs:
  - data/interim/brfss_2024_selected.csv.gz (gzip-compressed tabular data)
  - data/interim/brfss_2024_dictionary.json (metadata & codebook variable dictionary)
"""

import os
import sys
import time
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import pyreadstat

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "LLCP2024XPT.zip"
ARCHIVE_MEMBER_NAME = "LLCP2024.XPT "
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_CSV_GZ = INTERIM_DATA_DIR / "brfss_2024_selected.csv.gz"
OUTPUT_DICT_JSON = INTERIM_DATA_DIR / "brfss_2024_dictionary.json"

# Candidate Variable Specifications (Verified against CDC 2024 Codebook)
SELECTED_VARIABLES: Dict[str, Dict[str, Any]] = {
    # 1. Survey Design & Weighting
    "_STATE": {"category": "Survey Design", "role": "State FIPS identifier", "type": "int"},
    "_PSU": {"category": "Survey Design", "role": "Primary Sampling Unit cluster identifier", "type": "int"},
    "_STSTR": {"category": "Survey Design", "role": "Sample design stratification variable", "type": "int"},
    "_LLCPWT": {"category": "Survey Design", "role": "Final survey weight for landline and cellular combined", "type": "float"},

    # 2. Demographics & Socioeconomic Status
    "_SEX": {"category": "Demographics", "role": "Sex of respondent (1=Male, 2=Female)", "type": "categorical"},
    "_AGEG5YR": {"category": "Demographics", "role": "14-level five-year age categories (1=18-24 ... 13=80+, 14=Missing)", "type": "categorical"},
    "_AGE65YR": {"category": "Demographics", "role": "Binary age threshold (1=18-64, 2=65+, 3=Missing)", "type": "categorical"},
    "_AGE80": {"category": "Demographics", "role": "Imputed age value collapsed at 80 years old", "type": "numeric"},
    "_RACE": {"category": "Demographics", "role": "Race/ethnicity category (1=White NH ... 8=Hispanic, 9=Missing)", "type": "categorical"},
    "_EDUCAG": {"category": "Demographics", "role": "Education level completed (1=No HS ... 4=College Grad, 9=Missing)", "type": "categorical"},
    "_INCOMG1": {"category": "Demographics", "role": "Annual household income category (1=<$15k ... 7=$200k+, 9=Missing)", "type": "categorical"},
    "MARITAL": {"category": "Demographics", "role": "Marital status (1=Married, 2=Divorced, 3=Widowed ... 9=Refused)", "type": "categorical"},
    "EMPLOY1": {"category": "Demographics", "role": "Employment status (1=Employed ... 8=Unable to work, 9=Refused)", "type": "categorical"},
    "_METSTAT": {"category": "Demographics", "role": "Metropolitan status (1=Metropolitan, 2=Nonmetropolitan)", "type": "categorical"},
    "_URBSTAT": {"category": "Demographics", "role": "Urban/rural status (1=Urban, 2=Rural)", "type": "categorical"},

    # 3. General Health & Quality of Life
    "GENHLTH": {"category": "General Health", "role": "General health self-rating (1=Excellent ... 5=Poor, 7=DK, 9=Refused)", "type": "ordinal"},
    "_RFHLTH": {"category": "General Health", "role": "Calculated good or better health (1=Good/Better, 2=Fair/Poor, 9=Missing)", "type": "categorical"},
    "PHYSHLTH": {"category": "General Health", "role": "Number of days physical health not good in past 30 days (88=0 days)", "type": "count"},
    "_PHYS14D": {"category": "General Health", "role": "Physical health summary (1=0 days, 2=1-13 days, 3=14+ days, 9=Missing)", "type": "categorical"},
    "MENTHLTH": {"category": "General Health", "role": "Number of days mental health not good in past 30 days (88=0 days)", "type": "count"},
    "_MENT14D": {"category": "General Health", "role": "Mental health summary (1=0 days, 2=1-13 days, 3=14+ days, 9=Missing)", "type": "categorical"},
    "POORHLTH": {"category": "General Health", "role": "Number of days poor health prevented regular activities", "type": "count"},

    # 4. Healthcare Access
    "PRIMINS2": {"category": "Healthcare Access", "role": "Primary source of health insurance (88=None, 77=DK, 99=Refused)", "type": "categorical"},
    "_HLTHPL2": {"category": "Healthcare Access", "role": "Have any health care coverage (1=Yes, 2=No, 9=Missing)", "type": "categorical"},
    "PERSDOC3": {"category": "Healthcare Access", "role": "Have personal doctor/healthcare provider (1=Yes one, 2=More than one, 3=No)", "type": "categorical"},
    "MEDCOST1": {"category": "Healthcare Access", "role": "Could not see doctor due to cost in past 12 months (1=Yes, 2=No)", "type": "categorical"},
    "CHECKUP1": {"category": "Healthcare Access", "role": "Time since last routine checkup (1=Past year ... 8=Never)", "type": "ordinal"},

    # 5. Chronic Conditions (Candidate Targets & Comorbidities)
    "_MICHD": {"category": "Chronic Conditions", "role": "Reported coronary heart disease or myocardial infarction (1=Yes, 2=No)", "type": "target_candidate"},
    "CVDINFR4": {"category": "Chronic Conditions", "role": "Ever told had a heart attack / myocardial infarction (1=Yes, 2=No)", "type": "target_candidate"},
    "CVDCRHD4": {"category": "Chronic Conditions", "role": "Ever told had angina or coronary heart disease (1=Yes, 2=No)", "type": "target_candidate"},
    "CVDSTRK3": {"category": "Chronic Conditions", "role": "Ever told had a stroke (1=Yes, 2=No)", "type": "target_candidate"},
    "DIABETE4": {"category": "Chronic Conditions", "role": "Ever told had diabetes (1=Yes, 2=Pregnancy only, 3=No, 4=Pre-diabetes)", "type": "target_candidate"},
    "PREDIAB2": {"category": "Chronic Conditions", "role": "Ever told had pre-diabetes (1=Yes, 2=Pregnancy only, 3=No)", "type": "target_candidate"},
    "ASTHMA3": {"category": "Chronic Conditions", "role": "Ever told had asthma (1=Yes, 2=No)", "type": "target_candidate"},
    "_ASTHMS1": {"category": "Chronic Conditions", "role": "Computed asthma status (1=Current, 2=Former, 3=Never, 9=Missing)", "type": "categorical"},
    "CHCCOPD3": {"category": "Chronic Conditions", "role": "Ever told had C.O.P.D., emphysema, or chronic bronchitis (1=Yes, 2=No)", "type": "target_candidate"},
    "ADDEPEV3": {"category": "Chronic Conditions", "role": "Ever told had a depressive disorder (1=Yes, 2=No)", "type": "target_candidate"},
    "CHCKDNY2": {"category": "Chronic Conditions", "role": "Ever told had kidney disease (1=Yes, 2=No)", "type": "target_candidate"},
    "HAVARTH4": {"category": "Chronic Conditions", "role": "Ever told had arthritis / rheumatoid arthritis / gout / lupus (1=Yes, 2=No)", "type": "target_candidate"},
    "CHCSCNC1": {"category": "Chronic Conditions", "role": "Ever told had skin cancer (1=Yes, 2=No)", "type": "target_candidate"},
    "CHCOCNC1": {"category": "Chronic Conditions", "role": "Ever told had any other type of cancer (1=Yes, 2=No)", "type": "target_candidate"},

    # 6. Behavioral Risk Factors & Lifestyle
    "_BMI5": {"category": "Behaviors & Risk Factors", "role": "Computed Body Mass Index (continuous, 2 implied decimals)", "type": "continuous"},
    "_BMI5CAT": {"category": "Behaviors & Risk Factors", "role": "Four-category BMI (1=Underweight, 2=Normal, 3=Overweight, 4=Obese)", "type": "categorical"},
    "_RFBMI5": {"category": "Behaviors & Risk Factors", "role": "Overweight or obese calculated variable (1=No, 2=Yes, 9=Missing)", "type": "categorical"},
    "SMOKE100": {"category": "Behaviors & Risk Factors", "role": "Smoked at least 100 cigarettes in lifetime (1=Yes, 2=No)", "type": "categorical"},
    "_SMOKER3": {"category": "Behaviors & Risk Factors", "role": "Four-level smoker status (1=Daily, 2=Someday, 3=Former, 4=Never)", "type": "categorical"},
    "_RFSMOK3": {"category": "Behaviors & Risk Factors", "role": "Current cigarette smoking calculated variable (1=No, 2=Yes)", "type": "categorical"},
    "ECIGNOW3": {"category": "Behaviors & Risk Factors", "role": "Currently use e-cigarettes / vaping products (1=Daily, 2=Some days, 3=Not at all)", "type": "categorical"},
    "_CURECI3": {"category": "Behaviors & Risk Factors", "role": "Current e-cigarette user calculated status (1=No, 2=Yes)", "type": "categorical"},
    "EXERANY2": {"category": "Behaviors & Risk Factors", "role": "Participated in any physical activities in past 30 days (1=Yes, 2=No)", "type": "categorical"},
    "_TOTINDA": {"category": "Behaviors & Risk Factors", "role": "Adults with any leisure-time physical activity (1=Yes, 2=No)", "type": "categorical"},
    "DRNKANY6": {"category": "Behaviors & Risk Factors", "role": "Drank any alcoholic beverage in past 30 days (1=Yes, 2=No)", "type": "categorical"},
    "_RFBING6": {"category": "Behaviors & Risk Factors", "role": "Binge drinking calculated variable (1=No, 2=Yes)", "type": "categorical"},
    "_DRNKWK3": {"category": "Behaviors & Risk Factors", "role": "Calculated total drinks per week (continuous)", "type": "continuous"},
    "_RFDRHV9": {"category": "Behaviors & Risk Factors", "role": "Heavy drinkers calculated variable (1=No, 2=Yes)", "type": "categorical"},

    # 7. Functional Limitations & Disabilities
    "DEAF": {"category": "Disability & Function", "role": "Deaf or serious difficulty hearing (1=Yes, 2=No)", "type": "categorical"},
    "BLIND": {"category": "Disability & Function", "role": "Blind or serious difficulty seeing even with glasses (1=Yes, 2=No)", "type": "categorical"},
    "DECIDE": {"category": "Disability & Function", "role": "Difficulty concentrating, remembering, or making decisions (1=Yes, 2=No)", "type": "categorical"},
    "DIFFWALK": {"category": "Disability & Function", "role": "Serious difficulty walking or climbing stairs (1=Yes, 2=No)", "type": "categorical"},
    "DIFFDRES": {"category": "Disability & Function", "role": "Difficulty dressing or bathing (1=Yes, 2=No)", "type": "categorical"},
    "DIFFALON": {"category": "Disability & Function", "role": "Difficulty doing errands alone (1=Yes, 2=No)", "type": "categorical"},
}


def load_brfss_stream(
    archive_path: Path = RAW_ARCHIVE_PATH,
    member_name: str = ARCHIVE_MEMBER_NAME,
    columns: List[str] = list(SELECTED_VARIABLES.keys()),
    encoding: str = "latin1"
) -> pd.DataFrame:
    """
    Streams selected columns directly from the compressed XPT archive.

    Args:
        archive_path: Path to the raw ZIP archive.
        member_name: Exact entry name inside the archive.
        columns: List of variable names to extract.
        encoding: Character encoding for SAS XPT ('latin1').

    Returns:
        pd.DataFrame containing the selected columns for all survey records.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found at {archive_path}")

    logger.info(f"Opening archive: {archive_path} ({archive_path.stat().st_size:,} bytes)")
    t0 = time.time()

    with zipfile.ZipFile(archive_path, "r") as z:
        if member_name not in z.namelist():
            raise KeyError(f"Member '{member_name}' not found in archive. Available: {z.namelist()}")

        logger.info(f"Streaming member '{member_name.strip()}' with {len(columns)} requested columns...")
        with z.open(member_name) as f:
            df, meta = pyreadstat.read_xport(
                f,
                usecols=columns,
                encoding=encoding,
                output_format="pandas"
            )

    elapsed = time.time() - t0
    logger.info(f"Successfully streamed {len(df):,} rows and {len(df.columns)} columns in {elapsed:.2f}s.")
    return df


def validate_extracted_dataset(df: pd.DataFrame, expected_rows: int = 457670) -> None:
    """
    Validates data integrity, dimensions, and non-empty columns.
    """
    logger.info("Validating extracted dataset...")
    actual_rows, actual_cols = df.shape

    if actual_rows != expected_rows:
        logger.warning(
            f"Row count mismatch! Expected {expected_rows:,}, but got {actual_rows:,}. "
            f"Please verify with official CDC release counts."
        )
    else:
        logger.info(f"Row count validation PASSED: exactly {actual_rows:,} records.")

    expected_cols = set(SELECTED_VARIABLES.keys())
    actual_cols_set = set(df.columns)
    missing_cols = expected_cols - actual_cols_set

    if missing_cols:
        raise ValueError(f"Missing expected columns in extracted data: {missing_cols}")

    logger.info(f"Column count validation PASSED: all {len(expected_cols)} columns present.")

    # Check for completely empty columns
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        logger.warning(f"Columns with 100% missing values detected: {all_null_cols}")
    else:
        logger.info("Non-empty validation PASSED: No column is 100% null.")

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"In-memory DataFrame size: {mem_mb:.2f} MB.")


def export_interim_data(
    df: pd.DataFrame,
    output_csv_gz: Path = OUTPUT_CSV_GZ,
    output_dict_json: Path = OUTPUT_DICT_JSON
) -> None:
    """
    Saves the extracted dataset and variable dictionary into data/interim/.
    """
    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export gzip-compressed CSV
    logger.info(f"Saving gzip-compressed tabular data to {output_csv_gz}...")
    t0 = time.time()
    df.to_csv(output_csv_gz, index=False, compression="gzip")
    elapsed = time.time() - t0
    file_size_mb = output_csv_gz.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {output_csv_gz.name} ({file_size_mb:.2f} MB) in {elapsed:.2f}s.")

    # 2. Export variable dictionary JSON
    logger.info(f"Saving variable dictionary to {output_dict_json}...")
    dict_payload = {
        "dataset_name": "CDC BRFSS 2024 Selected Cohort",
        "source_archive": str(RAW_ARCHIVE_PATH.relative_to(PROJECT_ROOT)),
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_records": len(df),
        "total_variables": len(df.columns),
        "variables": SELECTED_VARIABLES
    }
    with open(output_dict_json, "w", encoding="utf-8") as f:
        json.dump(dict_payload, f, indent=2)
    logger.info("Variable dictionary saved successfully.")


def main() -> None:
    """Execution entry point."""
    logger.info("=== Starting CDC BRFSS 2024 Data Ingestion Pipeline (Milestone 3) ===")

    # 1. Stream data from compressed raw archive
    df = load_brfss_stream()

    # 2. Validate dimensions and completeness
    validate_extracted_dataset(df)

    # 3. Export to data/interim/
    export_interim_data(df)

    logger.info("=== Data Ingestion Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
