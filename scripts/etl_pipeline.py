"""Standalone ETL pipeline for the UCI Online Retail II dataset.

This script follows the M2 cleaning specification for the capstone project.
It can read the assignment's XLSX path, but it also falls back to the CSV or
ZIP that is actually present in this repository so the script runs end-to-end
without manual file changes.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NON_PRODUCT_STOCK_CODES = {
    "POST",
    "D",
    "C2",
    "M",
    "BANK CHARGES",
    "AMAZONFEE",
    "DCGSSBOY",
    "DCGSSGIRL",
    "gift_0001_10",
    "gift_0001_20",
    "gift_0001_30",
    "gift_0001_40",
    "gift_0001_50",
    "DOT",
}


def _resolve_existing_path(raw_path: str) -> Path:
    """Resolve the first existing raw data file path.

    Args:
        raw_path: The requested raw file path from the assignment.

    Returns:
        The first existing Path found among the assignment path and repo fallbacks.
    """
    requested = Path(raw_path)
    project_root = Path(__file__).resolve().parents[1]

    candidates = [
        requested,
        project_root / requested,
        requested.with_suffix(".csv"),
        requested.with_suffix(".zip"),
        project_root / requested.with_suffix(".csv"),
        project_root / requested.with_suffix(".zip"),
        project_root / "data" / "raw" / "online_retail_II.csv",
        project_root / "data" / "raw" / "online_retail_II.zip",
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find a raw dataset for {raw_path}. "
        "Checked XLSX, CSV, and ZIP fallbacks in the project."
    )


def load_data(raw_path: str) -> pd.DataFrame:
    """Load the raw dataset from Excel, CSV, or ZIP and return a dataframe.

    Args:
        raw_path: Assignment raw file path, usually data/raw/online_retail_II.xlsx.

    Returns:
        A pandas DataFrame containing the concatenated raw retail data.
    """
    resolved_path = _resolve_existing_path(raw_path)
    logger.info("Loading raw data from %s", resolved_path)

    suffix = resolved_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(resolved_path)
        sheet_names = workbook.sheet_names[:2]
        frames = [pd.read_excel(resolved_path, sheet_name=sheet_name) for sheet_name in sheet_names]
        df = pd.concat(frames, ignore_index=True)
        logger.info("Loaded workbook with sheets: %s", ", ".join(sheet_names))
        return df

    if suffix == ".zip":
        with zipfile.ZipFile(resolved_path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise FileNotFoundError("The ZIP archive does not contain a CSV file.")
            with archive.open(csv_members[0]) as file_obj:
                df = pd.read_csv(file_obj)
        logger.info("Loaded CSV file from ZIP member: %s", csv_members[0])
        return df

    df = pd.read_csv(resolved_path)
    logger.info("Loaded CSV file directly.")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the raw columns into snake_case names used across the project.

    Args:
        df: Raw dataframe with the original UCI column names.

    Returns:
        A copy of the dataframe with standardized snake_case column names.
    """
    renamed = df.rename(
        columns={
            "Invoice": "invoice_no",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_date",
            "Price": "unit_price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    ).copy()
    logger.info("Renamed columns to snake_case.")
    return renamed


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the invoice_date column to datetime values.

    Args:
        df: Dataframe with an invoice_date column.

    Returns:
        A dataframe where invoice_date is parsed to datetime.
    """
    result = df.copy()
    before_dtype = result["invoice_date"].dtype
    result["invoice_date"] = pd.to_datetime(result["invoice_date"], errors="coerce")
    after_dtype = result["invoice_date"].dtype
    logger.info("Parsed invoice_date from %s to %s.", before_dtype, after_dtype)
    logger.info(
        "Invoice date range: %s to %s",
        result["invoice_date"].min(),
        result["invoice_date"].max(),
    )
    return result


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where customer_id is missing.

    Args:
        df: Dataframe that may contain null customer_id values.

    Returns:
        A dataframe with null customer_id rows removed.
    """
    result = df.copy()
    before = len(result)
    result = result.loc[result["customer_id"].notna()].copy()
    removed = before - len(result)
    logger.warning("Removed %s rows with null customer_id.", removed)
    return result


def fix_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Convert customer_id to string and remove any trailing .0 suffix.

    Args:
        df: Dataframe with a customer_id column.

    Returns:
        A dataframe with customer_id normalized to string values.
    """
    result = df.copy()
    result["customer_id"] = (
        result["customer_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    logger.info("Normalized customer_id values to string format.")
    return result


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancelled invoices, identified by invoice_no values starting with C.

    Args:
        df: Dataframe with invoice_no values.

    Returns:
        A dataframe with cancelled orders removed.
    """
    result = df.copy()
    mask = result["invoice_no"].astype("string").str.startswith("C", na=False)
    removed = int(mask.sum())
    result = result.loc[~mask].copy()
    logger.warning("Removed %s cancellation rows.", removed)
    return result


def remove_invalid_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where quantity is less than or equal to zero.

    Args:
        df: Dataframe with a quantity column.

    Returns:
        A dataframe with invalid quantity rows removed.
    """
    result = df.copy()
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    mask = result["quantity"].le(0)
    removed = int(mask.sum())
    result = result.loc[~mask].copy()
    logger.warning("Removed %s rows with quantity <= 0.", removed)
    return result


def remove_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where unit_price is less than or equal to zero.

    Args:
        df: Dataframe with a unit_price column.

    Returns:
        A dataframe with invalid price rows removed.
    """
    result = df.copy()
    result["unit_price"] = pd.to_numeric(result["unit_price"], errors="coerce")
    mask = result["unit_price"].le(0)
    removed = int(mask.sum())
    result = result.loc[~mask].copy()
    logger.warning("Removed %s rows with unit_price <= 0.", removed)
    return result


def remove_non_products(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-product stock codes such as postage, fees, and manual lines.

    Args:
        df: Dataframe with a stock_code column.

    Returns:
        A dataframe with non-product stock codes removed.
    """
    result = df.copy()
    mask = result["stock_code"].astype("string").isin(NON_PRODUCT_STOCK_CODES)
    removed = int(mask.sum())
    result = result.loc[~mask].copy()
    logger.warning("Removed %s non-product stock_code rows.", removed)
    return result


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and convert description and country to uppercase.

    Args:
        df: Dataframe with description and country columns.

    Returns:
        A dataframe with standardized text columns.
    """
    result = df.copy()
    result["description"] = result["description"].astype("string").str.strip().str.upper()
    result["country"] = result["country"].astype("string").str.strip().str.upper()
    logger.info("Standardized description and country text columns.")
    return result


def add_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add a revenue column computed as quantity multiplied by unit_price.

    Args:
        df: Dataframe with numeric quantity and unit_price columns.

    Returns:
        A dataframe with a new revenue column.
    """
    result = df.copy()
    result["revenue"] = result["quantity"] * result["unit_price"]
    total_revenue = result["revenue"].sum()
    logger.info("Added revenue column. Total revenue: £%s", f"{total_revenue:,.2f}")
    return result


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from the cleaned dataframe.

    Args:
        df: Dataframe that may contain duplicate rows.

    Returns:
        A dataframe with duplicate rows removed.
    """
    result = df.copy()
    duplicates = int(result.duplicated().sum())
    result = result.drop_duplicates().copy()
    logger.warning("Removed %s duplicate rows.", duplicates)
    return result


def save_output(df: pd.DataFrame, output_path: str) -> None:
    """Save the cleaned dataframe to CSV.

    Args:
        df: The cleaned dataframe to write to disk.
        output_path: Destination CSV path.

    Returns:
        None.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    logger.info("Saved cleaned data to %s with %s rows.", destination, f"{len(df):,}")


def main() -> None:
    """Run the complete ETL pipeline end to end.

    Args:
        None.

    Returns:
        None.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_path = "data/raw/online_retail_II.xlsx"
    output_path = "data/processed/online_retail_cleaned.csv"

    df = load_data(raw_path)
    logger.info("Initial raw shape: %s rows x %s columns", f"{len(df):,}", len(df.columns))

    df = rename_columns(df)
    df = parse_dates(df)
    df = remove_nulls(df)
    df = fix_customer_id(df)
    df = remove_cancellations(df)
    df = remove_invalid_quantities(df)
    df = remove_invalid_prices(df)
    df = remove_non_products(df)
    df = clean_text_columns(df)
    df = add_revenue(df)
    df = remove_duplicates(df)

    logger.info("Final cleaned shape: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    save_output(df, output_path)


if __name__ == "__main__":
    main()
