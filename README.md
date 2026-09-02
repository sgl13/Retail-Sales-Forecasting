Retail Store Sales Forecasting Using Machine Learning and Time Series Analysis

📌 Project Overview

This project focuses on forecasting retail store sales using Machine Learning and Time Series Analysis.

The project is based on sales data from Corporación Favorita, a large Ecuadorian grocery retailer. The goal is to build a forecasting solution that predicts future sales for different product families across multiple stores.

The solution is intended to support:

Inventory planning

Supply chain optimization

Reduction of stock shortages

Data-driven business decisions

Domain: Retail Analytics
Difficulty Level: Advanced

🎯 Project Objectives

Understand the retail sales forecasting business problem

Clean and preprocess the datasets

Perform Exploratory Data Analysis (EDA)

Engineer meaningful time-series and business features

Build and compare multiple forecasting models

Evaluate model performance

Generate business insights

Develop an interactive dashboard

Deploy the forecasting solution

📂 Dataset

The project uses multiple datasets containing historical sales, product information, store information, holidays, transactions, oil prices, and prediction data.

Main Files

File

Description

train.csv

Historical training data containing date, store, product family, promotion information, and sales

test.csv

Data for the dates for which sales need to be predicted

sample_submission.csv

Sample file showing the required submission format

stores.csv

Store metadata including city, state, type, and cluster

oil.csv

Daily oil price information

holidays_events.csv

Holiday and event information

transactions.csv

Transaction-related data used as an additional business signal

submission.csv

Final generated prediction output

Important Fields

store_nbr – Identifies the store where products are sold.

family – Identifies the product family.

sales – Total sales for a product family at a particular store and date.

onpromotion – Number of items in a product family that were on promotion at a store on a particular date.

Dataset Link

Google Drive Dataset

🔎 Important Business Factors

Several external and calendar-related factors can influence retail sales.

Holidays and Events

The holidays_events.csv dataset contains information about holidays and events. Special attention is given to transferred holidays, bridge days, work days, and additional holidays.

Oil Prices

Ecuador's economy is influenced by oil prices, so daily oil price information can be considered as an external forecasting indicator.

Pay Days

Public-sector wages are paid every two weeks on the 15th and on the last day of the month. These dates may influence supermarket purchasing behavior.

Earthquake Effect

A magnitude 7.8 earthquake occurred in Ecuador on April 16, 2016. Relief efforts increased demand for products such as water and other essential goods for several weeks.

🔄 Project Workflow

Business Understanding
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Business Insights
        ↓
Dashboard Development
        ↓
Deployment

🧹 Data Cleaning

The data preparation stage includes:

Loading and inspecting datasets

Handling missing values

Checking duplicate records

Converting date columns into appropriate formats

Validating data types

Checking outliers and unusual sales values

Merging relevant datasets

Preparing data for time-series modeling

📊 Exploratory Data Analysis

The analysis covers:

Overall sales trends over time

Sales by store

Sales by product family

Promotion vs. sales relationship

Sales during holidays and events

Monthly and yearly sales patterns

Day-of-week sales patterns

Store-level performance

Product-family performance

Impact of oil prices

Impact of major events

🛠️ Feature Engineering

Date Features

Year

Month

Day

Day of week

Week of year

Quarter

Weekend indicator

Time-Series Features

Lag sales

Rolling averages

Rolling sums

Previous-day sales

Previous-week sales

Previous-month sales

Business Features

Store number

Product family

Promotion indicators

Holiday indicators

Store cluster

Store type

Oil price

Feature engineering is designed carefully to avoid data leakage, especially when creating lag and rolling features.

🤖 Machine Learning

The project evaluates multiple forecasting and machine-learning approaches:

LightGBM

XGBoost

CatBoost

Statistical time-series techniques

Weighted LightGBM + XGBoost Ensemble

The validation strategy respects the chronological order of the data rather than randomly shuffling observations.

📈 Model Evaluation

The models are evaluated using:

RMSLE

MAE

RMSE

R²

Model Comparison

Model

Evaluation Metric

Score

LightGBM

Validation L2 (MSE)

0.1861

XGBoost

Individual metric not recorded

—

LightGBM + XGBoost Ensemble

RMSLE

0.4127

LightGBM + XGBoost Ensemble

MAE

76.2772

LightGBM + XGBoost Ensemble

RMSE

262.7155

LightGBM + XGBoost Ensemble

R²

0.9554

Final Ensemble Configuration

LightGBM → 15%
XGBoost  → 85%

🏆 Best Model

Weighted LightGBM + XGBoost Ensemble

Validation performance:

RMSLE : 0.4127
MAE   : 76.2772
RMSE  : 262.7155
R²    : 0.9554

💡 Business Insights

The project provides insights into:

Which stores generate the highest sales

Which product families have the highest demand

How promotions affect sales

Seasonal demand patterns

The effect of holidays and events

The influence of external factors

Stores or products requiring additional inventory planning

📊 Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring sales trends, generating predictions, analyzing forecast distributions, and downloading prediction outputs.

Dashboard Capabilities

📈 Sales trend analysis

🏪 Store performance analysis

📦 Product-family analysis

🔮 Single sales prediction

📦 Batch prediction

📊 Forecast distribution

📏 Model performance metrics

📄 Prediction output download

🖥️ Dashboard Screenshots

The screenshot filenames below match the files currently present in the screenshots/ folder.

Dashboard Overview



🔮 Single Prediction



📦 Batch Prediction



📈 Sales Analytics



📄 Submission Prediction



ℹ️ About the Application



💻 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Plotly

Scikit-learn

XGBoost

LightGBM

CatBoost

Statsmodels

Streamlit

Git & GitHub

📁 Project Structure

Retail-Sales-Forecasting/
│
├── screenshots/
│   ├── About_me at 7.30.51 PM.png
│   ├── Analytic_dashboard 7.14.23 PM.png
│   ├── Analytics_Submission 2026-09-01 at 7.28.47 PM.png
│   ├── Dashboard_Preview at 7.10.55 PM.png
│   ├── Single_predictions at 7.17.49 PM.png
│   ├── Batch_predictions test 2026-09-01 at 7.21.00 PM.png
│   └── ...
│
├── notebooks/
│   └── RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb
│
├── scripts/
│   ├── check_pickle.py
│   ├── check_pickles.py
│   └── fix_metrics.py
│
├── app.py
├── main.py
├── predictions.py
├── requirements.txt
├── README.md
├── .gitignore
└── .python-version

🚀 How to Run the Project

1. Clone the repository

git clone https://github.com/sgl13/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

2. Create a virtual environment

python -m venv .venv

3. Activate the virtual environment

macOS / Linux:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

4. Upgrade pip

python -m pip install --upgrade pip

5. Install dependencies

pip install -r requirements.txt

6. Verify installation

python --version
streamlit --version

7. Run the Streamlit application

streamlit run app.py

Open the local URL shown in the terminal, usually:

http://localhost:8501

8. Stop the application

Press:

Ctrl + C

📓 Notebook

The main forecasting notebook is:

notebooks/
└── RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb

📦 Project Deliverables

Source Code

Jupyter Notebook

EDA

Feature Engineering

Trained Machine Learning Models

Interactive Streamlit Dashboard

Batch Prediction

Submission Prediction Output

Final Report

Presentation

GitHub Repository

🎓 Learning Outcomes

After completing this project, the learner can:

Execute an end-to-end data science project

Work with retail time-series data

Perform data cleaning and EDA

Engineer forecasting features

Build machine-learning forecasting models

Evaluate model performance

Develop interactive Streamlit dashboards

Generate business recommendations

Present a data science project using industry best practices

# 👨‍💻 Author

## Shivakumar G L

**Retail Sales Forecasting | Machine Learning | Time Series | Streamlit | AWS**

GitHub: [@sgl13](https://github.com/sgl13)

> **“Where predictions become interactive insights.”**

---

📌 Project Status

Status: ✅ Completed

The project includes data preprocessing, exploratory analysis, feature engineering, machine-learning model development, ensemble forecasting, model evaluation, Streamlit visualization, batch prediction, and submission output generation.
