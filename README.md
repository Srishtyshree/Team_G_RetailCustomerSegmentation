# NST DVA Capstone 2 - Project Repository

> **Newton School of Technology | Data Visualization & Analytics**
> A 2-week industry simulation capstone using Python, GitHub, and Tableau to convert raw data into actionable business intelligence.

---

## Before You Start

1. Rename the repository using the format `SectionName_TeamID_ProjectName`.
2. Fill in the project details and team table below.
3. Add the raw dataset to `data/raw/`.
4. Complete the notebooks in order from `01` to `05`.
5. Publish the final dashboard and add the public link in `tableau/dashboard_links.md`.
6. Export the final report and presentation as PDFs into `reports/`.

### Quick Start

If you are working locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

If you are working in Google Colab:

- Upload or sync the notebooks from `notebooks/`
- Keep the final `.ipynb` files committed to GitHub
- Export any cleaned datasets into `data/processed/`

---

## Project Overview

| Field | Details |
|---|---|
| **Project Title** | Customer Retention Strategy via RFM Segmentation — Online Retail II |
| **Sector** | Retail / E-Commerce |
| **Team ID** | _To be filled by team_ |
| **Section** | _To be filled by team_ |
| **Faculty Mentor** | _To be filled by team_ |
| **Institute** | Newton School of Technology |
| **Submission Date** | April 28, 2026 |

### Team Members

| Role | Name | GitHub Username |
|---|---|---|
| Project Lead | Gayatri Jaiswal | [`TechGenie-awake`](https://github.com/TechGenie-awake) |
| Data Lead | Gayatri Jaiswal | [`TechGenie-awake`](https://github.com/TechGenie-awake) |
| ETL Lead | Ananya | [`ANANYA542`](https://github.com/ANANYA542) |
| Analysis Lead | Mahek Motwani | [`MAHEK-ops`](https://github.com/MAHEK-ops) |
| Visualization Lead | Sneha Chepurwar | [`sneha31-debug`](https://github.com/sneha31-debug) |
| Strategy Lead | Srishty Shree | [`Srishtyshree`](https://github.com/Srishtyshree) |
| PPT and Quality Lead | Srishty Shree | [`Srishtyshree`](https://github.com/Srishtyshree) |

---

## Business Problem

A UK-based online gift retailer serves both individual shoppers and wholesale buyers across 30+ countries. Like most mid-sized e-commerce businesses, its marketing and retention spend is currently distributed evenly across the customer base — meaning low-value customers receive the same treatment as the small group driving most of the revenue, while high-value customers slip silently toward churn without targeted intervention. This project segments the active customer base using RFM (Recency, Frequency, Monetary) analysis to identify the segments worth protecting and the segments worth re-engaging, so the retention budget can be reallocated where it returns the most.

**Core Business Question**

> Which customer segments drive the most revenue, which are at the highest risk of churn, and where should retention spend be reallocated to recover lost value over the next 12 months?

**Decision Supported**

> Reallocation of retention and marketing campaign budget across customer segments — including which segments to reward, which to win back, and which to deprioritize — to reduce 12-month customer churn by 10–15% and lift repeat-purchase revenue from the top-quartile segment.

---

## Dataset

| Attribute | Details |
|---|---|
| **Source Name** | UCI Machine Learning Repository (donor: Daqing Chen, Sept 2019); accessed via Kaggle mirror `mashlyn/online-retail-ii-uci` |
| **Direct Access Link** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Row Count** | 1,067,371 |
| **Column Count** | 8 |
| **Time Period Covered** | December 1, 2009 – December 9, 2011 |
| **Format** | CSV (committed as 15 MB compressed ZIP at `data/raw/online_retail_II.zip`) |

**Key Columns Used**

| Column Name | Description | Role in Analysis |
|---|---|---|
| InvoiceNo | Six-digit transaction ID; `C` prefix = cancellation | Returns/cancellation analysis, transaction-level KPIs |
| InvoiceDate | Transaction timestamp | RFM Recency, time-series trends, dashboard filter |
| Quantity | Units purchased per transaction | Revenue computation, returns flagging |
| UnitPrice | Price per unit (GBP) | Revenue computation, RFM Monetary |
| CustomerID | Five-digit customer ID (~25% missing) | RFM segmentation, churn analysis (excluded if null) |
| Country | Customer's country of residence | Geographic dashboard filter, regional KPIs |
| StockCode | Five-digit product ID | Product-level analysis (excluded for non-product codes) |
| Description | Product name (free text) | Tableau filter, product-level reporting |

For full column definitions, see [`docs/data_dictionary.md`](docs/data_dictionary.md).

---

## KPI Framework

| KPI | Definition | Formula / Computation |
|---|---|---|
| Active Customer Count | Customers with ≥1 purchase in trailing 12 months | `nunique(CustomerID where InvoiceDate >= max_date - 365 days)` |
| Average Order Value (AOV) | Average revenue per invoice | `sum(Quantity * UnitPrice) / nunique(InvoiceNo)` |
| Customer Lifetime Value (CLV) | Total spend per customer over the dataset period | `sum(Quantity * UnitPrice) group by CustomerID` |
| 90-Day Churn Rate | Share of customers with no purchase in last 90 days | `count(CustomerID where recency_days > 90) / count(CustomerID)` |
| Revenue Concentration (Top 20%) | Revenue share of the top quintile of customers | `sum(revenue of top 20% of CustomerIDs by CLV) / sum(total revenue)` |
| Repeat Purchase Rate | Share of customers with ≥2 distinct invoices | `count(CustomerID with frequency >= 2) / count(CustomerID)` |
| Country Revenue Mix | Revenue share per country | `sum(revenue) group by Country / sum(total revenue)` |

Document KPI logic clearly in `notebooks/04_statistical_analysis.ipynb` and `notebooks/05_final_load_prep.ipynb`.

---

## Tableau Dashboard

| Item | Details |
|---|---|
| **Dashboard URL** | _Paste Tableau Public link here_ |
| **Executive View** | _Describe the high-level KPI summary view_ |
| **Operational View** | _Describe the detailed drill-down view_ |
| **Main Filters** | _List the interactive filters used_ |

Store dashboard screenshots in [`tableau/screenshots/`](tableau/screenshots/) and document the public links in [`tableau/dashboard_links.md`](tableau/dashboard_links.md).

---

## Key Insights

_List 8-12 major findings from the analysis, written in decision language. Each insight should tell the reader what to think or act upon, not merely describe a chart._

1. _Insight 1_
2. _Insight 2_
3. _Insight 3_
4. _Insight 4_
5. _Insight 5_
6. _Insight 6_
7. _Insight 7_
8. _Insight 8_

---

## Recommendations

_Provide 3-5 specific, actionable business recommendations, each linked directly to an insight above._

| # | Insight | Recommendation | Expected Impact |
|---|---|---|---|
| 1 | _Which insight does this address?_ | _What should the stakeholder do?_ | _What measurable impact do you expect?_ |
| 2 | _Which insight does this address?_ | _What should the stakeholder do?_ | _What measurable impact do you expect?_ |
| 3 | _Which insight does this address?_ | _What should the stakeholder do?_ | _What measurable impact do you expect?_ |

---

## Repository Structure

```text
SectionName_TeamID_ProjectName/
|
|-- README.md
|
|-- data/
|   |-- raw/                         # Original dataset (never edited)
|   `-- processed/                   # Cleaned output from ETL pipeline
|
|-- notebooks/
|   |-- 01_extraction.ipynb
|   |-- 02_cleaning.ipynb
|   |-- 03_eda.ipynb
|   |-- 04_statistical_analysis.ipynb
|   `-- 05_final_load_prep.ipynb
|
|-- scripts/
|   `-- etl_pipeline.py
|
|-- tableau/
|   |-- screenshots/
|   `-- dashboard_links.md
|
|-- reports/
|   |-- README.md
|   |-- project_report_template.md
|   `-- presentation_outline.md
|
|-- docs/
|   `-- data_dictionary.md
|
|-- DVA-oriented-Resume/
`-- DVA-focused-Portfolio/
```

---

## Analytical Pipeline

The project follows a structured 7-step workflow:

1. **Define** - Sector selected, problem statement scoped, mentor approval obtained.
2. **Extract** - Raw dataset sourced and committed to `data/raw/`; data dictionary drafted.
3. **Clean and Transform** - Cleaning pipeline built in `notebooks/02_cleaning.ipynb` and optionally `scripts/etl_pipeline.py`.
4. **Analyze** - EDA and statistical analysis performed in notebooks `03` and `04`.
5. **Visualize** - Interactive Tableau dashboard built and published on Tableau Public.
6. **Recommend** - 3-5 data-backed business recommendations delivered.
7. **Report** - Final project report and presentation deck completed and exported to PDF in `reports/`.

---

## Tech Stack

| Tool | Status | Purpose |
|---|---|---|
| Python + Jupyter Notebooks | Mandatory | ETL, cleaning, analysis, and KPI computation |
| Google Colab | Supported | Cloud notebook execution environment |
| Tableau Public | Mandatory | Dashboard design, publishing, and sharing |
| GitHub | Mandatory | Version control, collaboration, contribution audit |
| SQL | Optional | Initial data extraction only, if documented |

**Recommended Python libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`

---

## Evaluation Rubric

| Area | Marks | Focus |
|---|---|---|
| Problem Framing | 10 | Is the business question clear and well-scoped? |
| Data Quality and ETL | 15 | Is the cleaning pipeline thorough and documented? |
| Analysis Depth | 25 | Are statistical methods applied correctly with insight? |
| Dashboard and Visualization | 20 | Is the Tableau dashboard interactive and decision-relevant? |
| Business Recommendations | 20 | Are insights actionable and well-reasoned? |
| Storytelling and Clarity | 10 | Is the presentation professional and coherent? |
| **Total** | **100** | |

> Marks are awarded for analytical thinking and decision relevance, not chart quantity, visual decoration, or code length.

---

## Submission Checklist

**GitHub Repository**

- [ ] Public repository created with the correct naming convention (`SectionName_TeamID_ProjectName`)
- [ ] All notebooks committed in `.ipynb` format
- [ ] `data/raw/` contains the original, unedited dataset
- [ ] `data/processed/` contains the cleaned pipeline output
- [ ] `tableau/screenshots/` contains dashboard screenshots
- [ ] `tableau/dashboard_links.md` contains the Tableau Public URL
- [ ] `docs/data_dictionary.md` is complete
- [ ] `README.md` explains the project, dataset, and team
- [ ] All members have visible commits and pull requests

**Tableau Dashboard**

- [ ] Published on Tableau Public and accessible via public URL
- [ ] At least one interactive filter included
- [ ] Dashboard directly addresses the business problem

**Project Report**

- [ ] Final report exported as PDF into `reports/`
- [ ] Cover page, executive summary, sector context, problem statement
- [ ] Data description, cleaning methodology, KPI framework
- [ ] EDA with written insights, statistical analysis results
- [ ] Dashboard screenshots and explanation
- [ ] 8-12 key insights in decision language
- [ ] 3-5 actionable recommendations with impact estimates
- [ ] Contribution matrix matches GitHub history

**Presentation Deck**

- [ ] Final presentation exported as PDF into `reports/`
- [ ] Title slide through recommendations, impact, limitations, and next steps

**Individual Assets**

- [ ] DVA-oriented resume updated to include this capstone
- [ ] Portfolio link or project case study added

---

## Contribution Matrix

This table must match evidence in GitHub Insights, PR history, and committed files.

_Initial planned ownership shown below — will be finalized in Phase 2 based on actual GitHub commit + PR history._

| Team Member | Dataset and Sourcing | ETL and Cleaning | EDA and Analysis | Statistical Analysis | Tableau Dashboard | Report Writing | PPT and Viva |
|---|---|---|---|---|---|---|---|
| Gayatri Jaiswal (M1) | Owner | Support | — | — | — | Support | Support |
| Ananya (M2) | Support | Owner | Support | — | — | — | — |
| Mahek Motwani (M3) | — | Support | Owner | Support | — | — | — |
| Sneha Chepurwar (M4) | — | — | Support | Owner | Owner | — | — |
| Srishty Shree (M5) | — | — | — | Support | Support | Owner | Owner |

_Declaration: We confirm that the above contribution details are accurate and verifiable through GitHub Insights, PR history, and submitted artifacts._

**Team Lead Name:** Gayatri Jaiswal

**Date:** _______________

---

## Academic Integrity

All analysis, code, and recommendations in this repository must be the original work of the team listed above. Free-riding is tracked via GitHub Insights and pull request history. Any mismatch between the contribution matrix and actual commit history may result in individual grade adjustments.

---

*Newton School of Technology - Data Visualization & Analytics | Capstone 2*
