# Phase 1 Proposal — DVA Capstone 2

**Sector:** Retail / E-Commerce
**Team ID:** G
**Team Members:**
- Gayatri Jaiswal — M1 Project Lead ([`TechGenie-awake`](https://github.com/TechGenie-awake))
- Ananya — M2 Data Engineer ([`ANANYA542`](https://github.com/ANANYA542))
- Mahek Motwani — M3 EDA & KPI Analyst ([`MAHEK-ops`](https://github.com/MAHEK-ops))
- Sneha Chepurwar — M4 Statistician & Dashboard Builder ([`sneha31-debug`](https://github.com/sneha31-debug))
- Srishty Shree — M5 Insights, Report & Presentation Lead ([`Srishtyshree`](https://github.com/Srishtyshree))

**Submission window:** Phase 1
**Institute:** Newton School of Technology

---

## 1. Problem Statement

> **For the marketing and retention head of a UK-based online gift retailer, segment the active customer base using RFM (Recency, Frequency, Monetary) analysis to identify the top revenue-driving segments and the customers at highest risk of churn, so retention spend and personalized campaigns can be reallocated toward segments with the highest projected lifetime value — with a target of reducing 12-month customer churn by 10–15% and lifting repeat-purchase revenue from the top-quartile segment.**

## 2. Why This Problem Matters

In e-commerce, **retaining a customer costs roughly 5× less than acquiring a new one**, and the top 20% of customers typically drive 60–80% of revenue (Pareto effect). For a mid-sized retailer, even a small reduction in churn translates directly into recurring revenue. Most retailers, however, treat all customers identically in their marketing spend — losing both money on low-value segments and high-value customers to silent churn.

A data-backed segmentation strategy lets the retention team:
- Concentrate spend where it returns the most
- Identify "silent churn" customers before they're lost
- Tailor campaigns by segment maturity and value tier

## 3. Decision-Maker

**Primary decision-maker:** Head of Marketing & Customer Retention
**Secondary stakeholders:** CMO, CRM/Email Marketing Lead, Country Sales Managers

## 4. Scope

### In Scope
- Customer-level RFM segmentation
- Churn-risk scoring at the customer level
- Country-level revenue and customer mix
- Top-product analysis per segment
- Retention recommendations with estimated revenue impact

### Out of Scope
- Demographic profiling (no demographic data in source)
- Real-time recommendation engine
- Pricing strategy or markdown optimization
- Supply chain / inventory planning
- Acquisition channel attribution (no acquisition data)

## 5. Success Criteria

| Dimension | Criterion |
|---|---|
| **Statistical rigor** | RFM segments are statistically distinct; optional K-Means clusters validated via silhouette score |
| **Dashboard usability** | Tableau dashboard supports filtering by segment, country, and time period |
| **Business clarity** | 3–5 prioritized retention actions, each tied to an estimated £ impact |
| **Reproducibility** | End-to-end pipeline runs from raw zip → cleaned data → KPIs → dashboard extract |

## 6. Initial KPI Framework

| KPI | Definition | Why it matters |
|---|---|---|
| **Active Customer Count** | Customers with ≥1 purchase in trailing 12 months | Baseline retention pool |
| **Average Order Value (AOV)** | Total revenue ÷ total orders | Spend depth signal |
| **Customer Lifetime Value (CLV)** | Total spend per customer over period | Identifies high-value segments |
| **90-Day Churn Rate** | % of customers with no purchase in last 90 days | Silent churn early-warning |
| **Revenue Concentration (Top 20%)** | % of revenue from top-quintile customers | Pareto validation |
| **Repeat Purchase Rate** | % of customers with ≥2 orders | Loyalty signal |
| **Country Revenue Mix** | Revenue share per country | Geographic concentration risk |

## 7. Dataset Selection

### Primary Dataset

**Name:** UCI Online Retail II
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (donated by Daqing Chen, September 2019)
**Accessed via:** Kaggle mirror — `mashlyn/online-retail-ii-uci`
**Size:** 1,067,371 rows × 8 columns
**Period:** December 1, 2009 – December 9, 2011 (≈ 2 years)
**Format:** CSV (committed as a 15 MB compressed ZIP in `data/raw/online_retail_II.zip`)
**Domain:** UK-based online gift retailer; retail + wholesale customers

**Why this dataset fits:**
- Real, transactional, and unaggregated — supports customer-level analysis
- Contains realistic data quality issues (missing CustomerIDs, returns, cancellations, free items) — strong ETL story
- 2-year horizon enables both segmentation and time-based churn modelling
- Multi-country footprint enables geographic dashboard slicing
- Recognized academic dataset — strong viva defensibility

**Capstone 1 reuse check:** ✅ This dataset was **not** used in Capstone 1.

### Backup Dataset 1

**Name:** Brazilian E-Commerce (Olist)
**Source:** Kaggle — `olistbr/brazilian-ecommerce`
**Size:** 100k+ orders across 9 joinable CSV tables
**Why a backup:** Same retail domain, more complex multi-table joins. Strong fallback if a wider data model is desired.

### Backup Dataset 2

**Name:** Amazon Product Reviews (UCSD — Julian McAuley)
**Source:** UCSD — listed in NST approved sources
**Size:** Configurable per category subset (100k–500k+ rows)
**Why a backup:** Adds review/sentiment dimension if the team wants to pivot toward review-driven product analysis.

## 8. Anticipated Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Mentor rejects primary dataset | Backup 1 (Olist) is ready and pre-checked against the same row/column rules |
| ~25% missing CustomerIDs | These rows are excluded from RFM analysis; quantified and reported in cleaning log |
| 2-year window may be too short for cohort analysis | Use rolling 6-month windows for trend stability |
| Large file size | Already mitigated by committing the 15 MB compressed ZIP rather than the 94 MB raw CSV |

## 9. Deliverables Map (Phase 1 → Phase 2)

| Phase 1 (this submission) | Phase 2 (final) |
|---|---|
| Problem statement ✅ | Cleaned dataset in `data/processed/` |
| Dataset selection ✅ | All 5 notebooks committed |
| Initial KPI framework ✅ | `scripts/etl_pipeline.py` |
| Backup datasets identified ✅ | Tableau Public dashboard (with interactive filter) |
| Repo skeleton committed ✅ | `reports/project_report.pdf` + `reports/presentation.pdf` |
