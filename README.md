# Retail Store Sales Forecasting

## 📌 Project Overview

This project focuses on analyzing retail sales data and building a **Machine Learning and Time Series Forecasting solution** to predict future sales.

The project is based on the **Corporación Favorita Grocery Sales Forecasting dataset**, which contains historical sales information from multiple stores and product families in Ecuador.

The solution combines:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Time Series Analysis
5. Machine Learning
6. Ensemble Modeling
7. Model Evaluation
8. Sales Prediction
9. Interactive Streamlit Dashboard
10. AWS SageMaker Deployment Preparation

---

## 🎯 Project Objective

The main objective is to build a forecasting system that can predict future retail sales and provide useful business insights.

The solution can help with:

* Inventory planning
* Demand forecasting
* Supply-chain optimization
* Promotion planning
* Stock shortage reduction
* Store-level decision making
* Product-family demand analysis

---

## 📊 Dataset

The project uses the **Corporación Favorita Grocery Sales Forecasting dataset**.

### Main Datasets

| Dataset                 | Description                             |
| ----------------------- | --------------------------------------- |
| `train.csv`             | Historical sales data used for training |
| `test.csv`              | Future dates for generating predictions |
| `stores.csv`            | Store information                       |
| `oil.csv`               | Daily oil price information             |
| `holidays_events.csv`   | Holiday and event information           |
| `transactions.csv`      | Store transaction information           |
| `sample_submission.csv` | Required prediction submission format   |
| `submission.csv`        | Generated prediction output             |

### Important Features

| Feature       | Description                     |
| ------------- | ------------------------------- |
| `date`        | Sales date                      |
| `store_nbr`   | Store identifier                |
| `family`      | Product family                  |
| `sales`       | Sales quantity                  |
| `onpromotion` | Number of products on promotion |

---

## 🔍 Exploratory Data Analysis

The EDA focuses on understanding sales patterns and important business factors.

### Analysis Includes

* Checking missing values
* Checking duplicate records
* Data type validation
* Sales distribution
* Sales trends over time
* Store-level sales analysis
* Product-family analysis
* Promotion vs. sales analysis
* Holiday and event analysis
* Monthly and yearly sales patterns
* Day-of-week sales patterns
* Oil price analysis
* Major event analysis

### Business Factors

Important external factors considered include:

* Holidays
* Promotions
* Pay days
* Oil prices
* Store characteristics
* Product families
* Seasonal demand
* Major events such as the 2016 Ecuador earthquake

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

* Loading multiple datasets
* Inspecting dataset structure
* Handling missing values
* Removing duplicate records
* Converting date columns
* Validating data types
* Merging related datasets
* Checking unusual sales values
* Preparing chronological data for forecasting

Special care is taken to avoid **data leakage** when creating time-series features.

---

## 🛠️ Feature Engineering

Several date, time-series, and business features are created.

### 📅 Date Features

* Year
* Month
* Day
* Day of Week
* Week of Year
* Quarter
* Weekend Indicator

### ⏱️ Time-Series Features

* Lag Sales
* Previous Day Sales
* Previous Week Sales
* Previous Month Sales
* Rolling Mean
* Rolling Sum

### 🏪 Business Features

* Store Number
* Product Family
* Promotion Indicator
* Holiday Indicator
* Store Type
* Store Cluster
* Oil Price

> ⚠️ Time-series features are created carefully to prevent future information from leaking into the training data.

---

## 🤖 Machine Learning

The project evaluates multiple machine-learning and forecasting approaches.

### Models Used

* LightGBM
* XGBoost
* CatBoost
* Statistical Time-Series Techniques
* Weighted LightGBM + XGBoost Ensemble

The validation strategy respects the chronological order of the data instead of randomly shuffling observations.

---

## 📈 Model Evaluation

The models are evaluated using forecasting metrics such as:

* RMSLE
* MAE
* RMSE
* R²

### Model Comparison

| Model                       | Metric              |        Score |
| --------------------------- | ------------------- | -----------: |
| LightGBM                    | Validation L2 (MSE) |   **0.1861** |
| XGBoost                     | Individual metric   |            — |
| LightGBM + XGBoost Ensemble | RMSLE               |   **0.4127** |
| LightGBM + XGBoost Ensemble | MAE                 |  **76.2772** |
| LightGBM + XGBoost Ensemble | RMSE                | **262.7155** |
| LightGBM + XGBoost Ensemble | R²                  |   **0.9554** |

### Final Ensemble

The final forecasting model uses a weighted ensemble:

```text
LightGBM → 15%
XGBoost  → 85%
```

### 🏆 Final Model

**Weighted LightGBM + XGBoost Ensemble**

Validation performance:

```text
RMSLE : 0.4127
MAE   : 76.2772
RMSE  : 262.7155
R²    : 0.9554
```

---

## 📊 Interactive Streamlit Dashboard

The project includes an interactive **Streamlit dashboard** for exploring sales and generating predictions.

### Dashboard Features

* 📈 Sales trend analysis
* 🏪 Store performance analysis
* 📦 Product-family analysis
* 🔮 Single sales prediction
* 📦 Batch prediction
* 📊 Forecast distribution
* 📏 Model performance metrics
* 📄 Prediction output download

### Dashboard Screenshots

#### Dashboard Overview

![Dashboard Overview](screenshots/dashboard_overview%20at%207.26.22%E2%80%AFPM.png)

#### Single Prediction

![Single Prediction](screenshots/single_prediction%20at%207.28.26%E2%80%AFPM.png)

#### Batch Prediction

![Batch Prediction](screenshots/batch_prediction_test%20at%207.32.38%E2%80%AFPM.png)

#### Sales Analytics

![Sales Analytics](screenshots/analytics_dashboard_test2%20at%207.36.47%E2%80%AFPM.png)

#### Submission Prediction

![Submission Prediction](screenshots/submission_prediction%20at%207.34.05%E2%80%AFPM.png)

---

## 📁 Project Structure

```text
Retail-Sales-Forecasting/
│
├── notebooks/
│   └── RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb
│
├── scripts/
│   ├── check_pickle.py
│   ├── check_pickles.py
│   └── fix_metrics.py
│
├── screenshots/
│   ├── dashboard_overview.png
│   ├── single_prediction.png
│   ├── batch_prediction.png
│   ├── analytics_dashboard.png
│   └── submission_prediction.png
│
├── app.py
├── main.py
├── predictions.py
├── requirements.txt
├── README.md
├── .gitignore
└── .python-version
```

### Ignored Files

The following files and directories are intentionally excluded from GitHub:

```text
data/
*.csv
models/*.pkl
.idea/
.venv/
venv/
__pycache__/
.DS_Store
```

This keeps large datasets, trained model artifacts, virtual environments, and local IDE files outside the repository.

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/sgl13/Retail-Sales-Forecasting.git
```

Move into the project directory:

```bash
cd Retail-Sales-Forecasting
```

---

## 2. Create a virtual environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

---

## 3. Activate the virtual environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

After activation, your terminal should show something similar to:

```text
(.venv)
```

---

## 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 5. Install project dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

## 6. Verify the installation

You can verify Streamlit with:

```bash
streamlit --version
```

You can also check Python:

```bash
python --version
```

---

## 7. Run the Streamlit application

Start the dashboard using:

```bash
streamlit run app.py
```

The terminal will display a local URL similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser:

```text
http://localhost:8501
```

---

## 8. Stop the application

To stop Streamlit, return to the terminal and press:

```text
Ctrl + C
```

---

## 📓 Notebooks

The project contains the main forecasting notebook:

### Retail Sales Forecasting Notebook

```text
notebooks/
└── RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb
```

The notebook covers:

* Data understanding
* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Time-series analysis
* Model development
* Model evaluation
* Ensemble forecasting
* Prediction generation

The notebook can be opened using **Jupyter Notebook, JupyterLab, or PyCharm**.

---

# 💡 Business Insights

The forecasting solution can be used to identify:

* Highest-performing stores
* Highest-demand product families
* Promotion impact
* Seasonal demand patterns
* Holiday effects
* Store-level demand differences
* Product-level demand patterns
* Potential inventory requirements

These insights can support better:

* Inventory planning
* Supply-chain decisions
* Promotional strategies
* Demand planning
* Store operations

---

# ☁️ AWS SageMaker Deployment

The project is being prepared for deployment using **Amazon SageMaker**.

### Planned Architecture

```text
GitHub
   │
   ▼
Amazon S3
   │
   ▼
SageMaker Training Job
   │
   ▼
Trained Model
   │
   ▼
SageMaker Model
   │
   ▼
SageMaker Endpoint
   │
   ▼
Real-Time Prediction
```

### Planned Deployment Components

* Training script
* Inference script
* Model artifact
* Amazon S3 storage
* SageMaker training job
* SageMaker model
* SageMaker endpoint
* Endpoint testing

### Current Status

```text
Core ML Project     → Completed ✅
Streamlit Dashboard → Completed ✅
GitHub Repository   → Completed ✅
SageMaker           → In Progress 🚧
```

---

# 💻 Technologies Used

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Statsmodels

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Application

* Streamlit

### Version Control

* Git
* GitHub

### Cloud

* Amazon S3
* Amazon SageMaker

---

# 🎓 Skills Demonstrated

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Time-Series Forecasting
* Machine Learning
* Ensemble Learning
* Model Evaluation
* Retail Analytics
* Data Visualization
* Streamlit Application Development
* Git & GitHub
* AWS Cloud Deployment

---

# 📦 Project Deliverables

* ✅ Source Code
* ✅ Jupyter Notebook
* ✅ Data Cleaning
* ✅ Exploratory Data Analysis
* ✅ Feature Engineering
* ✅ Machine Learning Models
* ✅ Ensemble Forecasting
* ✅ Model Evaluation
* ✅ Streamlit Dashboard
* ✅ Single Prediction
* ✅ Batch Prediction
* ✅ Prediction Output
* 🚧 AWS SageMaker Deployment

---

# 📌 Project Status

**Status: 🚧 Deployment Phase**

The core retail sales forecasting solution has been completed, including data preprocessing, EDA, feature engineering, machine-learning model development, ensemble forecasting, evaluation, and Streamlit dashboard development.

The next phase is deploying the forecasting model using **AWS SageMaker**.

---

# 👨‍💻 Author

## Shivakumar G L

**Retail Sales Forecasting | Machine Learning | Time Series | Streamlit | AWS**

GitHub: [@sgl13](https://github.com/sgl13)

> **“Where predictions become interactive insights.”**

---
