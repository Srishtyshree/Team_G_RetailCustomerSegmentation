# Data Dictionary

## Dataset Summary

| Item | Details |
|---|---|
| Dataset name | UCI Online Retail II |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (donated by Daqing Chen, September 2019) |
| Accessed via | Kaggle mirror — `mashlyn/online-retail-ii-uci` |
| Raw file name | `data/raw/online_retail_II.csv` and `data/raw/online_retail_II.zip` |
| Time period | December 1, 2009 – December 9, 2011 |
| Rows | 1,067,371 |
| Columns | 8 |
| Granularity | One row per line-item per invoice |
| Domain | UK-based online gift retailer (retail + wholesale customers) |

## Column Definitions

| Column Name | Data Type | Description | Example Value | Used In | Cleaning Notes |
|---|---|---|---|---|---|
| InvoiceNo | string | Six-digit transaction identifier | `536365` | EDA / KPI / Filter | Prefix `C` indicates a cancellation — exclude from sales KPIs |
| StockCode | string | Five-digit product identifier | `85123A` | EDA / Tableau | Some non-numeric codes (e.g. `POST`, `M`) for postage/manual lines — flag and exclude from product KPIs |
| Description | string | Product name (free text) | `WHITE HANGING HEART T-LIGHT HOLDER` | Tableau filter | Inconsistent casing and trailing whitespace — strip and uppercase-normalize |
| Quantity | integer | Units purchased per transaction | `6` | KPI / RFM | Negative = return; treat separately in returns analysis |
| InvoiceDate | datetime | Transaction timestamp | `2009-12-01 07:45:00` | EDA / RFM (Recency) | Cast to datetime; derive `year`, `month`, `day_of_week` |
| UnitPrice | float | Price per unit, British pounds (GBP) | `2.55` | KPI / RFM (Monetary) | `0` = free / promotional item — flag and review |
| CustomerID | string | Five-digit customer identifier | `17850` | RFM / Segmentation | ~25% missing — exclude null rows from customer-level analysis |
| Country | string | Customer country of residence | `United Kingdom` | Tableau filter / Geographic KPIs | UK-dominant (~91% of rows); group small countries as "Other" if needed |

## Derived Columns

| Derived Column | Logic | Business Meaning |
|---|---|---|
| `revenue` | `Quantity * UnitPrice` | Line-item revenue contribution |
| `is_return` | `Quantity < 0` | Boolean flag for returns |
| `is_cancelled` | `InvoiceNo.startswith('C')` | Boolean flag for cancelled invoices |
| `recency_days` | `(reference_date - max(InvoiceDate)) per CustomerID` | RFM Recency score input |
| `frequency` | `count(unique InvoiceNo) per CustomerID` | RFM Frequency score input |
| `monetary` | `sum(revenue) per CustomerID` | RFM Monetary score input |
| `rfm_score` | Concatenation of R, F, M scores (1–5 each) | Customer segment label |

## Data Quality Notes

- **Missing CustomerID** — ~25% of rows. These are excluded from all customer-level analyses (RFM, segmentation, churn). Quantified in the cleaning log.
- **Cancellations** — InvoiceNo starting with `C` indicates a cancelled order. Treated separately in returns/cancellation analysis; excluded from sales KPIs.
- **Returns** — Negative `Quantity` values. Used in returns-rate KPI; excluded from RFM monetary computation.
- **Free items** — `UnitPrice = 0`. Inspected and flagged as promotional or data-entry artifacts.
- **Non-product line items** — `StockCode` values like `POST`, `M`, `D`, `BANK CHARGES` represent postage, manual adjustments, discounts. Excluded from product-level analysis.
- **Description inconsistencies** — Casing, whitespace, and synonyms. Normalized via stripping and uppercase.
- **Geographic skew** — UK accounts for the majority of transactions; international markets are smaller and more volatile. Note this when interpreting country-level segments.

For full transformation log, see [`notebooks/02_cleaning.ipynb`](../notebooks/02_cleaning.ipynb).
