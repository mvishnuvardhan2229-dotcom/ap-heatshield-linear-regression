# AP HeatShield: Andhra Pradesh Heatwave & Peak Temperature Forecasting Using Linear Regression

> **Course Assignment:** Introduction to Data Science (IDS)  
> **Author:** Machabathuni Vishnu Vardhan  
> **Live Demo:** [View Streamlit Web Application](https://ap-heatshield-linear-regression-vishnu.streamlit.app)

---

## 📌 Project Overview

Most data science students rely on pre-cleaned, readymade CSV datasets from Kaggle. For this Introduction to Data Science project, I wanted to build an **end-to-end pipeline from scratch**—starting by extracting raw, messy meteorological web data, cleaning broken sensor artifacts, engineering time-series features, training a Multiple Linear Regression model, and deploying it as an interactive web app.

**AP HeatShield** focuses on Vijayawada (Gannavaram meteorological station) in Andhra Pradesh. It ingests historical daily weather observations to estimate tomorrow's peak afternoon temperature ($TM$) and evaluate outdoor heat-stress risk levels.

---

## 🔄 End-to-End Workflow Followed

1. **Web Scraping:** Extracted multi-year historical monthly climate archives directly from public HTML tables using Python.
2. **Data Diagnostics & Cleaning:** Inspected missing values, dropped columns with high sensor failure rates (`>40%`), removed HTML footer summary artifacts, and applied **Time-Series Linear Interpolation**.
3. **Feature Engineering:** Developed autoregressive lags, moving averages, and cyclical seasonal encodings (`sin`/`cos`) while avoiding multicollinearity.
4. **Rigorous Model Training:** Trained a Multiple Linear Regression model using a strict chronological split (preventing data leakage) across a full 365-day unseen test cycle.
5. **Deployment:** Packaged the trained model (`.pkl`) and cleaned dataset into an interactive **Streamlit** dashboard deployed on Cloud infrastructure.

---

## 🌐 Web Scraping Details

To ensure the scraper was polite and reliable without violating server limits:
* **Source:** TuTiempo Global Climate Network / NOAA meteorological archive (`ws-431810` - Vijayawada Gannavaram Station).
* **Technique:** Used Python's `requests` library combined with `BeautifulSoup` to parse HTML DOM structure (`<table class="medias mensuales numspan">`).
* **Polite Scraping:** Implemented controlled delays (`time.sleep(0.5)` to `0.8`) between requests to avoid overloading the server.
* **URL Pattern:** Structured clean monthly loops (`MM-YYYY`) to fetch records seamlessly from 2022 through 2025.

---

## 📊 Dataset & Distribution Details

* **Total Records:** 1,332 daily observations spanning 4 full years (2022–2025).
* **Target Variable ($y$):** `target_next_tm` (Tomorrow's Maximum Temperature in °C).
* **Dropped Feature:** Sea Level Atmospheric Pressure (`SLP`) was dropped automatically during diagnostics because it exceeded 40% missing sensor records.
* **Active Input Features ($X$):**
  * `TM`: Today's Maximum Temperature (°C)
  * `T`: Daily Mean Temperature (°C)
  * `Tm`: Minimum Morning Temperature (°C)
  * `H`: Relative Humidity (%)
  * `PP`: Precipitation / Rainfall (mm)
  * `VV`: Visibility (km)
  * `V`: Average Wind Speed (km/h)
  * `tm_lag1`: Yesterday's Maximum Temperature
  * `tm_roll3`: 3-Day Rolling Average Max Temperature
  * `tm_roll7`: 7-Day Rolling Average Max Temperature
  * `h_roll3`: 3-Day Rolling Average Humidity
  * `season_sin` / `season_cos`: Cyclical Day-of-Year Seasonality Encodings

---

## 📈 Evaluation & Honest Reporting (Test-Set Results)

To ensure academic honesty, I did **not** shuffle the dataset randomly (which would be "time-travel cheating" by letting the model see future dates during training). Instead, I trained on past years and reserved a full **365-day unseen test cycle**.

* **Train $R^2$ Score:** `0.9188` (91.88%)
* **Test $R^2$ Score:** `0.9032` (90.32%)
* **Test Mean Absolute Error (MAE):** `0.69 °C` *(The model predicts tomorrow's peak temperature within less than 0.7°C of error on average).*
* **Test Root Mean Squared Error (RMSE):** `0.96 °C`

### Learned Linear Regression Feature Weights
| Feature | Coefficient Weight | Interpretation |
| :--- | :--- | :--- |
| `T` | `+0.6395` | Strong positive baseline with daily mean temperature. |
| `season_sin` | `+0.5178` | Captures smooth seasonal transition across summer/winter cycles. |
| `TM` | `+0.3390` | Today's peak heat anchors tomorrow's maximum temperature. |
| `tm_roll7` | `+0.1720` | Weekly thermal momentum. |
| `tm_lag1` | `-0.1177` | Acts as a built-in rate-of-change correction when paired with `TM`. |

---

## ⚠️ Limitations

1. **Atmospheric Complexity:** Weather is inherently chaotic. While Linear Regression captures macro-seasonal trends and daily thermal inertia exceptionally well ($R^2 = 0.903$), it cannot anticipate sudden micro-meteorological anomalies like localized cloudbursts or abrupt wind-direction shifts driven by Bay of Bengal depressions.
2. **Station Specificity:** The model is trained strictly on Vijayawada station data; applying it directly to hilly or coastal zones (like Araku or Visakhapatnam Beach Road) would require localized station retraining.

---

## 💡 Key Learning Outcomes

* Learned how to inspect raw HTML tables, write resilient scrapers, and diagnose dirty data (`'-'` strings, footer summary rows).
* Mastered **Time-Series Linear Interpolation** for filling missing sensor gaps without distorting seasonal trends.
* Understood the severe impact of **Multicollinearity** in Linear Regression and how removing redundant mathematical difference columns improves model stability.
* Avoided data leakage by enforcing chronological train-test splits rather than random shuffling.

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/ap-heatshield-linear-regression.git]
   cd ap-heatshield-linear-regression
   
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
