# Assignment 13: Data Gathering, Preprocessing & Exploratory Data Analysis (EDA)

**Author / Student:** Gururaj Khule  
**Course:** GenAI / Machine Learning Engineering  
**Role:** Junior Machine Learning Engineer  
**Submission Folder:** `GenAI-Task13-GururajKhule`  
**Notebook File:** `Assignment_13_Data_Gathering_Preprocessing_EDA.ipynb`  

---

## 1. Problem Statement & Scope

As a Junior Machine Learning Engineer, before building or training any predictive Machine Learning model, the core responsibility is to:
1. **Gather Data from Diverse Sources:** Enterprise data is distributed across flat files (CSV), semi-structured formats (JSON), relational database management systems (SQLite), and external web services (REST APIs).
2. **Understand & Preprocess the Data:** Clean corrupt or incomplete records, handle missing values with sound statistical justification, eliminate duplicates, fix inconsistent naming conventions (standardizing to lowercase `snake_case`), correct data types, and encode categorical features for mathematical processing.
3. **Conduct Exploratory Data Analysis (EDA):** Perform systematic Univariate and Bivariate analysis across all standard Matplotlib and Seaborn visualization types to extract actionable patterns, identify outliers, evaluate feature correlations, and formulate engineering hypotheses.

### Restrictions Followed:
- **Strictly used:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Requests`, and `SQLite3` only.
- **No model training:** Strictly focused on pre-ML foundations.
- **Code clarity:** Every single step is thoroughly commented with its engineering and mathematical logic.

---

## 2. Directory Structure

```text
GenAI-Task13-GururajKhule/
│
├── Assignment_13_Data_Gathering_Preprocessing_EDA.ipynb  # Primary Jupyter Notebook with all 10 tasks
├── Readme.md                                             # Comprehensive documentation & logic guide
├── titanic.csv                                           # Kaggle Titanic CSV dataset (Tasks 1, 5, 6, 7, 8, 9, 10)
├── products.json                                         # Semi-structured JSON catalog (Task 2)
├── sample.db                                             # SQLite relational database with employees table (Task 3)
└── tmdb_movies.csv                                       # TMDB popular & top-rated movie dataset (Task 4)
```

---

## 3. Detailed Task-by-Task Implementation & Logic Breakdown

### PART 1 — Data Gathering (Multiple Sources)

#### Task 1: Load Data from CSV
- **Dataset:** Kaggle Titanic Dataset (`titanic.csv`).
- **Function:** `pd.read_csv("titanic.csv")`.
- **Inspections & Logic:**
  - `shape`: Dimensionality check ($894$ rows $\times 12$ columns).
  - `columns`: Verification of schema naming and consistency.
  - `head()`, `tail()`, `sample()`: Inspecting boundary rows and random records to identify missing values and formatting anomalies.
  - `info()`: Non-null counts, exact memory usage ($83.9+\text{ KB}$), and storage data types.
  - `describe()` & `describe(include=['O'])`: Parametric metrics for numerical features (mean, std, percentiles) and frequency metrics for text categories.
  - `isnull().sum()` & `duplicated().sum()`: Baseline data quality health check.

#### Task 2: Load Data from JSON
- **Dataset:** `products.json` containing e-commerce items (product IDs, names, categories, pricing, stock levels, ratings, and tag arrays).
- **Function:** `pd.read_json("products.json")`.
- **Logic:** Semi-structured data (nested JSON objects and arrays) is converted into a 2D tabular DataFrame, enabling vectorized filtering and aggregations (e.g. mean price per category).

#### Task 3: Load Data from SQL Database
- **Database:** Local SQLite database (`sample.db`).
- **Table Schema:** `employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL)`.
- **Logic:**
  1. Connected via Python's standard `sqlite3` library.
  2. Executed DDL to create table and DML to seed 8 diverse employee records.
  3. Ingested data into Pandas using `pd.read_sql_query("SELECT * FROM employees", conn)`.
  4. Demonstrated analytical SQL aggregation (`GROUP BY department`) calculating mean and max salaries.
  5. Closed database connection safely using `conn.close()`.

#### Task 4: API Mini Project (TMDB API — Movie Data Collector)
- **Service:** The Movie Database (TMDB) REST API (`/movie/popular` & `/movie/top_rated`).
- **Fields Extracted:** `title`, `release_date`, `vote_average` (Rating), `popularity`, `vote_count`, `original_language`, `overview`, `category`.
- **Logic:**
  - Designed a production-grade collector function using Python `requests` sending HTTP `GET` requests with query parameters (`api_key`, `language`, `page`).
  - Added an offline fallback to cached authentic TMDB movie data (`tmdb_movies.csv`) to guarantee seamless execution even without an internet connection or personal API key.
  - Exported collected movie records to `tmdb_movies.csv`.

---

### PART 2 — Data Preprocessing & Cleaning (Kaggle Dataset)

#### Task 5: Understanding the Data
- **Shape & Dtypes:** 894 initial rows, 12 features (7 numerical, 5 categorical).
- **Programmatic Identification:** Automated separation using `df.select_dtypes(include=[np.number])` and `df.select_dtypes(include=['object'])`.
- **Missing Value Audit:**
  - `Cabin`: 689 missing ($77.07\%$).
  - `Age`: 177 missing ($19.80\%$).
  - `Embarked`: 2 missing ($0.22\%$).
- **Additional Lecture Techniques:**
  - **Cardinality Check:** Evaluated unique values via `nunique()`.
  - **Skewness Analysis:** Calculated skewness coefficients. Found `Fare` heavily right-skewed ($4.79$) and `Age` moderately skewed ($0.39$).
  - **IQR Outlier Detection:** Applied $Q_1 - 1.5 \times \text{IQR}$ and $Q_3 + 1.5 \times \text{IQR}$ to quantify outlier percentages across numerical features (`Fare`: $12.98\%$, `Age`: $1.53\%$).

#### Task 6: Data Cleaning
- **1. Missing Value Imputation Strategy:**
  - **Numerical (`Age`):** Imputed with the **median** ($28.0$ years).  
    *Logic:* Because `Age` contains outliers and right skew, the median is robust to extreme values, unlike the mean ($29.7$ years) which would introduce bias.
  - **Categorical (`Embarked`):** Imputed with the **mode** (`'S'`).  
    *Logic:* With only 2 missing entries out of 894, the most frequent port represents $>72\%$ of passengers and preserves distribution without distortion.
  - **Categorical (`Cabin`):** Imputed with explicit category **`'Unknown'`**.  
    *Logic:* Over $77\%$ of records are missing. Imputing mode would create massive false concentration. Labeling as `'Unknown'` preserves the predictive signal that missing cabin correlates with lower-deck ticket holders.
- **2. Duplicate Row Removal:**
  - Detected 3 duplicate records (`df.duplicated().sum() == 3`).
  - Removed with `df.drop_duplicates(inplace=True)` and reset index, bringing clean row count to exactly 891 rows.
- **3. Standardizing Column Names (lowercase snake_case):**
  - Mapped PascalCase names to PEP 8 standard lowercase snake_case:  
    `PassengerId` $\rightarrow$ `passenger_id`, `SibSp` $\rightarrow$ `sib_sp`, `Parch` $\rightarrow$ `parch`, etc.
- **4. Fixing Data Types:**
  - Converted `age` to rounded float, `fare` to 2-decimal float, `pclass` and `survived` to explicit integers.

#### Task 7: Feature Preparation (Basic Encoding & Feature-Target Separation)
- **Categorical Encoding Strategy:**
  - **Binary Label Encoding on `sex`:**  
    Mapped `{'male': 0, 'female': 1}`. Binary encoding is optimal because it preserves compact 1D vectorization without creating unnecessary sparse columns.
  - **One-Hot Encoding on `embarked`:**  
    Applied `pd.get_dummies(..., prefix='embarked', drop_first=True)`.  
    *Logic:* Embarkation port is nominal (no intrinsic order). Setting `drop_first=True` drops the reference level to avoid the **Dummy Variable Trap** (multicollinearity where $D_1 + D_2 + D_3 = 1$).
- **Feature and Target Separation:**
  - **Target Vector ($y$):** `survived` ($891$ binary labels).
  - **Feature Matrix ($X$):** $891$ rows $\times 8$ clean predictor features (`pclass`, `age`, `sib_sp`, `parch`, `fare`, `sex_encoded`, `embarked_Q`, `embarked_S`), dropping non-predictive identifiers (`passenger_id`, `name`, `ticket`, `cabin`).

---

### PART 3 — Exploratory Data Analysis (EDA)

#### Task 8: Univariate Analysis
Rendered all visualization types covered in Matplotlib and Seaborn lectures:
1. **Histogram + KDE (`sns.histplot(kde=True)`):** Age distribution with overlaid mean ($29.7$) and median ($28.0$) reference lines.
2. **Histogram + KDE for Fare:** Capturing right-skewed revenue distribution.
3. **Count Plot for Class (`sns.countplot`):** Class breakdown showing $55\%$ of passengers traveled in 3rd class.
4. **Count Plot for Survival:** Class balance showing $61.5\%$ mortality vs $38.5\%$ survival.
5. **Box Plot for Fare (`sns.boxplot`):** Outlier identification highlighting luxury ticket fares ($>\$500$).
6. **Violin Plot for Age (`sns.violinplot`):** Kernel density estimate combined with internal quartile lines.
7. **Pie Chart for Embarkation Port (`plt.pie`):** Visualizing passenger origin ($72.4\%$ Southampton, $18.9\%$ Cherbourg, $8.7\%$ Queenstown).
8. **ECDF + Rug Plot (`sns.ecdfplot` + `sns.rugplot`):** Empirical cumulative distribution of passenger age with discrete tick annotations.

#### Task 9: Bivariate Analysis
Comprehensive bivariate charts examining inter-feature interactions:
1. **Numerical vs. Numerical:**
   - **Scatter Plot with Survival Hue:** `age` vs `fare` segmented by survival status.
   - **Linear Regression Plot (`sns.regplot`):** Best-fit trendline with confidence interval shading.
   - **Correlation Heatmap (`sns.heatmap`):** Masked Pearson correlation matrix highlighting relationships between class, fare, and survival.
   - **Pairplot (`sns.pairplot`):** Multi-panel grid of numerical distributions and pairwise scatters conditioned on `survived`.
2. **Categorical vs. Numerical:**
   - **Bar Plot with Error Bars (`sns.barplot`):** Mean ticket fare by passenger class with $95\%$ confidence intervals.
   - **Box Plot (`sns.boxplot`):** Class-wise fare dispersion and outlier spread.
   - **Violin Plot with Split Hue (`sns.violinplot(split=True)`):** Age density across survival conditioned by gender.
   - **Strip Plot (`sns.stripplot`):** Jittered individual data points illustrating fare distribution across classes.
   - **Interaction Point Plot (`sns.pointplot`):** Survival probability trends across classes differentiated by gender.
3. **Categorical vs. Categorical:**
   - **Grouped Count Plot (`sns.countplot`):** Survival count comparison between male and female passengers.
   - **Contingency Matrix Heatmap (`pd.crosstab` + `sns.heatmap`):** Normalized percentage survival rate matrix across passenger classes.

---

## 4. Key Insights & Observations (Task 10)

1. **Gender Survival Disparity ("Women and Children First"):**  
   Female survival rate reached **$74.2\%$**, compared to only **$18.9\%$** for males. In bivariate count and point plots, gender is the single strongest predictor of survival.
2. **Socioeconomic Hierarchy & Survival Privilege:**  
   1st class passengers experienced a **$>62\%$** survival probability, 2nd class **$\sim 47\%$**, and 3rd class only **$\sim 24\%$**. Passenger class exhibits a strong negative correlation with mortality ($r \approx -0.34$).
3. **Fare Distribution Outliers & Heavy Skewness:**  
   The `fare` variable exhibits extreme positive skewness ($> 4.5$). The median fare is $\$14.45$, whereas luxury suites exceeded $\$500.00$. Downstream linear/distance-based models will require non-linear log transformations (`np.log1p`).
4. **Demographic Dynamics (Age Cohorts):**  
   Infants and young children ($< 10$ years) had elevated survival probability across all ticket classes. Working-age adult males in 3rd class suffered the highest mortality of any demographic group.
5. **Data Quality & Pipeline Hygiene:**  
   Missing value imputation (median for skewed continuous `age`, mode for nominal `embarked`, and explicit `'Unknown'` for high-missingness `cabin`) combined with duplicate removal and snake_casing established a clean, leak-free feature matrix $X$ and target vector $y$.

---

## 5. How to Run the Notebook

1. Ensure Python 3.8+ is installed with standard dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn requests
   ```
2. Open `Assignment_13_Data_Gathering_Preprocessing_EDA.ipynb` in:
   - VS Code (with the Jupyter extension), or
   - Jupyter Notebook / Jupyter Lab (`jupyter notebook`), or
   - Google Colaboratory.
3. Run all cells sequentially via **Kernel $\rightarrow$ Restart and Run All**. All data dependencies (`titanic.csv`, `products.json`, `sample.db`, `tmdb_movies.csv`) reside in the same folder.
