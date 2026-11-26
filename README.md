# 🚢 Cruise Price Prediction

Predicting cruise ticket prices from itinerary, timing and company features, using a web-scraped dataset, feature engineering and regression models (Linear Regression vs. Random Forest).

---

## 📌 Project Overview

This project builds an end-to-end data science pipeline for cruise pricing:

1. **Scrape** cruise listings from [center.cruises](https://center.cruises/cruises-sea/) with Python.
2. **Clean** the raw data (duplicates, missing prices, Russian-language text and dates).
3. **Engineer features** from dates, itinerary text and company-level statistics.
4. **Explore** the data with statistics, hypothesis tests and visualizations.
5. **Model** the price with Linear Regression (baseline) and Random Forest.
6. **Visualize** key figures in a Tableau dashboard.

## 🎯 Objectives

- Collect a structured cruise dataset from a public listing website.
- Identify which factors (company, duration, season, region) relate to cruise prices.
- Predict cruise price from trip and company-related features.
- Compare a linear baseline with a non-linear ensemble model.

## 📊 Dataset Description

| Item | Details |
|---|---|
| Source | https://center.cruises/cruises-sea/ (scraped with `requests` + `BeautifulSoup`) |
| Raw scrape | 20,000 rows × 7 columns (`name`, `ship`, `company`, `price`, `duration`, `start_date`, `end_date`) |
| After cleaning | 9,999 rows after duplicate removal; **5,053 rows** after removing listings with `price ≤ 10` |
| Final file | `powerBI_dataset.csv` — **5,053 rows × 32 columns** |
| Price range | €11 – €999 (mean ≈ €665.5, median €699) |
| Duration | 1 – 244 nights (median 7) |
| Departure dates | 2026-04-25 to 2027-04-10 |
| Companies | 9 (MSC Cruises, Royal Caribbean International, Costa Cruises, Norwegian Cruise Lines, Celebrity Cruises, MSC Yacht Club, Azamara Cruises, The Ritz-Carlton Yacht Collection, Oceania Cruises) |
| Target | `price_log = log1p(price)` |

**Column groups**
- **Original:** `name`, `ship`, `company`, `price`, `duration`, `start_date`, `end_date`
- **Itinerary:** `destinations`, `first_stop`, `last_stop`, `num_ports`, `region`, `name_len`
- **Pricing:** `price_per_night`, `price_segment`, `price_log`
- **Time:** `departure_month`, `departure_day_of_week`, `is_weekend`, `season`, `month_sin`, `month_cos`
- **Company/ship:** `avg_price_by_company`, `avg_price_by_season`, `avg_price_by_region`, `fleet_size`, `company_segment`, `ship_occurrence_freq`, `is_premium_company`
- **Encoded:** `company_label_enc`, `season_label_enc`, `company_segment_label_enc`

> ⚠️ **Data notes:** prices are in euros (€). The maximum price is exactly €999, and some luxury brands (Oceania, Azamara, Ritz-Carlton) have very low listed prices, which may reflect a different pricing basis on the source site. Missing years in dates were filled with 2026.

## 🛠 Technologies and Libraries Used

- **Language:** Python (developed in Google Colab)
- **Scraping:** `requests`, `beautifulsoup4`, `logging`
- **Data handling:** `pandas`, `numpy`
- **Translation:** `deep-translator` (Google Translate)
- **Statistics:** `scipy`
- **ML:** `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `LabelEncoder`, metrics)
- **Visualization:** `matplotlib`, `seaborn`, `plotly`, `squarify`
- **Dashboard:** Tableau (`.twb`)

## 🔄 Project Workflow

### 1. Data Collection
- Paginated scraper over `https://center.cruises/cruises-sea/page-{n}/` (up to 1,000 pages, stops at the first empty page).
- Browser-like `User-Agent`, 3 retries per page, 10 s timeout, 0.5 s delay between requests, and rotating log file.
- Each listing card yields name, ship, company, date range, duration and price.
- `requests` + `BeautifulSoup` were used instead of Selenium because the data is present in the static HTML.

### 2. Data Cleaning and Preprocessing
- Split the `date` field into `start_date` / `end_date`; removed weekday prefixes, translated Russian month abbreviations to English, added the year where missing and converted to `datetime`.
- Extracted numeric values from `price` and `duration` (e.g. `"7 ночей"` → `7`).
- Removed the `"Лайнер: "` prefix from `ship`.
- Translated cruise names from Russian to English with `deep-translator`.
- Dropped duplicates (`keep='last'`): 20,000 → 9,999 rows.
- Filled 94 missing prices with the **median**.
- Cleaned `name` with regex (removed durations, cruise codes, brackets) to get `destinations`, then derived `first_stop` and `last_stop`.
- Removed listings with `price ≤ 10` (placeholder prices) before modelling.

### 3. Exploratory Data Analysis
- Descriptive statistics, skewness/kurtosis and a normality test for `price`.
- Group-by comparisons (region, company, season × price segment) and top/bottom-N records.
- Hypothesis tests (see [Key Findings](#-key-findings)).
- 14 visualizations: histogram, bar, box, Spearman heatmap, line, violin, donut, scatter, pair plot, treemap, three interactive Plotly charts and a summary panel.

### 4. Feature Engineering
| Feature(s) | How it was built |
|---|---|
| `price_per_night`, `price_segment` | `price / duration`; Budget (< 100), Mid-range (100–250), Luxury (> 250) |
| `departure_month`, `departure_day_of_week`, `is_weekend`, `season` | Extracted from `start_date` |
| `month_sin`, `month_cos` | Cyclical encoding of the month |
| `num_ports`, `name_len` | Comma-separated item count and character length of the cruise name |
| `region` | Keyword matching on the name (USA, Mediterranean, Northern Europe, Middle East, Other) |
| `avg_price_by_company / season / region` | Group means of `price_per_night` mapped back to each row |
| `fleet_size` | Unique ships per company |
| `company_segment` | Mainstream / Premium / Luxury from the company's average price per night |
| `ship_occurrence_freq` | Number of listings per ship |
| `is_premium_company` | 1 if the company is in a hand-picked premium list |
| `*_label_enc` | `LabelEncoder` on `company`, `season`, `company_segment` |
| `price_log` | `log1p(price)` to reduce right skew |

### 5. Model Development
- **Target:** `price_log`
- **Features (16):** `duration`, `num_ports`, `departure_month`, `departure_day_of_week`, `is_weekend`, `fleet_size`, `ship_occurrence_freq`, `is_premium_company`, `month_sin`, `month_cos`, `avg_price_by_company`, `avg_price_by_season`, `avg_price_by_region`, `company_label_enc`, `season_label_enc`, `company_segment_label_enc`
- **Split:** 80 / 20 (4,042 train / 1,011 test), `random_state=42`
- **Models:** `LinearRegression` (baseline) and `RandomForestRegressor(n_estimators=100, random_state=42)`
- No feature scaling or hyperparameter tuning was applied.

### 6. Model Evaluation
RMSE, MAE and R² were computed on the **held-out test set**, on the **log-transformed target** (not on euro prices). Predictions are converted back to euros with `np.expm1`.

## 📈 Model Performance and Comparison

Test-set metrics (log scale):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 0.54 | 0.36 | 0.3324 |
| **Random Forest** | **0.16** | **0.10** | **0.9391** |

Random Forest is clearly the better model, which suggests the price relationship is non-linear.

> 📝 **Note:** metrics on the original euro scale are not reported in the notebook. A re-computation with the same split gave roughly RMSE ≈ €91, MAE ≈ €63, R² ≈ 0.83 for Random Forest. Log-scale scores should not be read as euro-scale accuracy.

Example prediction (7-night, 5-port, June weekend departure, MSC-like profile): **≈ €650**.

## 🔍 Key Findings

- **Company matters:** one-way ANOVA shows mean price differs significantly across the 9 companies (F = 165.5, p ≈ 1e-248). Duration also differs by company (F = 278.7).
- **Duration vs. price:** a weak *negative* Pearson correlation (r = −0.12, p ≈ 2e-18), so longer cruises are not more expensive in total price.
- **Segments differ:** Welch t-tests show significant price differences between Luxury and Mid-range segments (p ≈ 6e-10) and between Summer and Winter departures (p ≈ 2e-22).
- **Market mix:** Mediterranean (34%) and USA (33%) dominate; 7-night cruises are the most common length; most listings are Mid-range (64%) or Budget (34%), with Luxury about 2%.
- **Modelling:** Random Forest (R² 0.94 on log price) far outperforms Linear Regression (R² 0.33). The log transform was used to handle right-skewed prices.

## 📉 Dashboard Overview

`powerBI_dashboard.twb` is a **Tableau** workbook (despite the file name). Its **Dashboard 1** contains:

- **KPIs** — average price, average duration and record count.
- **AVG Price by company** — bar chart.
- **AVG duration by company** — chart by company.

It also contains an empty worksheet ("Sheet 5"). The workbook references `cruiseclean.csv` on a local path, so re-point the data source to `powerBI_dataset.csv` after opening it in Tableau Desktop.

## 📁 Repository Structure

```
├── powerBI_code.ipynb          # Scraping, cleaning, feature engineering, EDA, ML
├── powerBI_dataset.csv         # Final dataset (5,053 rows × 32 columns)
├── powerBI_dashboard.twb       # Tableau dashboard workbook
├── powerBI_presentation.pdf    # Project presentation
├── powerBI_report.pdf          # Detailed project report
└── README.md
```

## ▶️ How to Run the Project

No `requirements.txt` is included; the dependencies below come from the notebook's imports.

```bash
pip install requests beautifulsoup4 pandas numpy deep-translator scikit-learn scipy matplotlib seaborn plotly squarify
```

1. Open `powerBI_code.ipynb` in Google Colab or Jupyter.
2. **Scraping and cleaning cells** hit the live website and write intermediate CSVs (`cruises_raw.csv`, `cruises_translated.csv`, `cruises_cleaned.csv`). Paths use `/content/`, so adjust them for local use. The site may have changed since collection.
3. **To skip scraping**, load `powerBI_dataset.csv` and run from the "Exploratory Data Analysis" and "Machine Learning" sections.
4. Open `powerBI_dashboard.twb` in Tableau Desktop and re-link the data source.

> ⚠️ **Known limitations:** some cells use names that are not defined elsewhere in the notebook (`f_oneway`, `monthly_price`) and the data-loading cell references `cruises_cleaned (2).csv`, so a top-to-bottom run needs small fixes. The report and presentation also contain figures (e.g. dataset size and some test statistics) that differ from the notebook.

## ✅ Conclusion

Random Forest predicts log cruise price much better than a linear baseline (R² 0.94 vs 0.33 on the test set). Company identity, fleet size and trip duration are the main signals. Limitations include a modest dataset (5,053 listings from one site), group-mean features computed before the train/test split, the lack of cross-validation or tuning, and listings that repeat routes across dates. Future work could include cross-validation, hyperparameter tuning, splitting by route or ship to avoid optimistic scores, and reporting error in euros.
