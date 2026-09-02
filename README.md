<div align="center">

🛒 Retail Store Sales Forecasting

Machine Learning & Time Series Forecasting

An end-to-end retail analytics project that transforms historical sales data into actionable demand forecasts.

<br>







</div>

⭐ Why This Project?

Retail businesses need accurate demand forecasts to decide what to stock, where to stock it, and when to stock it.

This project builds a complete forecasting pipeline using historical retail sales from Corporación Favorita, an Ecuadorian grocery retailer. It combines data cleaning, exploratory analysis, time-series feature engineering, machine learning, ensemble modeling, and an interactive Streamlit application.

Business Outcomes

📦 Inventory Planning

🚚 Supply Chain

📊 Decision Making

Better demand estimation

Improved allocation planning

Data-driven business insights

Reduce stock shortages

Store-level planning

Identify high-demand products

🎯 Project Snapshot

Category

Details

Domain

Retail Analytics

Problem

Store & product-family sales forecasting

Data Source

Corporación Favorita retail sales data

Approach

Machine Learning + Time Series Features

Best Model

Weighted LightGBM + XGBoost Ensemble

Ensemble Weight

LightGBM 15% + XGBoost 85%

Best RMSLE

0.4127

R²

0.9554

Application

Interactive Streamlit Dashboard

Status

🟢 Completed

🚀 Key Features

<table>
<tr>
<td width="50%" valign="top">

🔮 Prediction

Single sales prediction

Batch prediction

Submission prediction

Prediction output download

</td>
<td width="50%" valign="top">

📊 Analytics

Sales trend analysis

Store performance analysis

Product-family analysis

Forecast distribution

Model performance metrics

</td>
</tr>
</table>

🧠 Solution Architecture

                    RETAIL SALES DATA
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Sales Data        Store Data        External Data
   train.csv         stores.csv        oil.csv
   test.csv                           holidays_events.csv
   transactions.csv
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                  DATA PREPROCESSING
                           ↓
                       EDA
                           ↓
                 FEATURE ENGINEERING
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      Date Features   Lag Features    Business Features
          │                │                │
          └────────────────┼────────────────┘
                           ↓
                   MODEL DEVELOPMENT
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
       LightGBM         XGBoost          CatBoost
          │                │
          └────────┬───────┘
                   ↓
            WEIGHTED ENSEMBLE
          LightGBM 15% + XGBoost 85%
                   ↓
             SALES FORECAST
                   ↓
          STREAMLIT DASHBOARD

📂 Dataset

The project uses multiple datasets containing historical sales, product information, store information, holidays, transactions, oil prices, and prediction data.

Dataset Files

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

Important Features

Feature

Meaning

store_nbr

Identifies the store

family

Identifies the product family

sales

Total sales for a product family at a store and date

onpromotion

Number of items in a product family on promotion

📥 Dataset

Access the Dataset on Google Drive

🔍 Data Analysis

🧹 Data Cleaning

The data preparation pipeline includes:

Loading and inspecting datasets

Handling missing values

Checking duplicate records

Converting date columns

Validating data types

Checking outliers and unusual sales values

Merging relevant datasets

Preparing data for time-series modeling

📊 Exploratory Data Analysis

The analysis covers:

Sales & Products

Time & Events

External Factors

Overall sales trends

Monthly/yearly patterns

Oil prices

Sales by store

Day-of-week patterns

Holidays & events

Sales by product family

Seasonal demand

Major events

Promotion vs. sales

Store performance

Earthquake effect

Product-family performance

Demand patterns

—

🛠️ Feature Engineering

📅 Date Features

Year • Month • Day • Day of Week • Week of Year • Quarter • Weekend Indicator

⏱️ Time-Series Features

Lag Sales • Rolling Averages • Rolling Sums • Previous-Day Sales • Previous-Week Sales • Previous-Month Sales

🏪 Business Features

Store Number • Product Family • Promotion Indicators • Holiday Indicators • Store Cluster • Store Type • Oil Price

⚠️ Feature engineering is designed carefully to avoid data leakage, particularly when generating lag and rolling features.

🌍 Important Business Factors

<table>
<tr>
<td width="50%" valign="top">

🎉 Holidays & Events

The holidays_events.csv dataset contains information about holidays and events, including transferred holidays, bridge days, work days, and additional holidays.

🛢️ Oil Prices

Daily oil prices are included as an external forecasting indicator because Ecuador's economy is influenced by oil prices.

</td>
<td width="50%" valign="top">

💰 Pay Days

Public-sector wages are paid every two weeks on the 15th and the last day of the month. These dates may influence supermarket purchasing behavior.

🌍 Earthquake Effect

A magnitude 7.8 earthquake occurred in Ecuador on April 16, 2016. Relief efforts increased demand for products such as water and other essential goods for several weeks.

</td>
</tr>
</table>

🤖 Machine Learning

The project evaluates multiple forecasting and machine-learning approaches:

Model

Purpose

LightGBM

Gradient boosting model for efficient tabular forecasting

XGBoost

Gradient boosting model for strong predictive performance

CatBoost

Additional machine-learning comparison

Statistical Time-Series Techniques

Traditional forecasting comparison

Weighted Ensemble

Combines LightGBM and XGBoost predictions

Validation Strategy

The validation strategy respects the chronological order of the data rather than randomly shuffling observations, which is important for time-series forecasting.

📈 Model Performance

🏆 Final Model

Weighted LightGBM + XGBoost Ensemble

Metric

Result

RMSLE

0.4127

MAE

76.2772

RMSE

262.7155

R²

0.9554

Ensemble Configuration

Model

Weight

LightGBM

15%

XGBoost

85%

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

💡 Business Insights

The project is designed to answer practical retail questions:

<table>
<tr>
<td width="50%" valign="top">

Which stores generate the highest sales?

Which product families have the highest demand?

How do promotions affect sales?

What are the seasonal demand patterns?

</td>
<td width="50%" valign="top">

How do holidays and events affect demand?

What influence do external factors have?

Which stores or products require additional inventory planning?

</td>
</tr>
</table>

🖥️ Interactive Streamlit Dashboard

The Streamlit application brings the forecasting model and analytics together in an interactive interface.

Dashboard Modules

Module

Purpose

🏠 Dashboard Overview

Application overview and key information

🔮 Single Prediction

Generate an individual sales forecast

📦 Batch Prediction

Generate predictions for multiple records

📈 Sales Analytics

Explore trends and forecast distributions

📄 Submission Prediction

View and analyze generated prediction data

ℹ️ About Application

Project and application information

📸 Dashboard Preview

<table>
<tr>
<td width="50%" valign="top">

🏠 Dashboard Overview

<img src="screenshots/Dashboard_Preview%20at%207.10.55%20PM.png" width="100%">

</td>
<td width="50%" valign="top">

🔮 Single Prediction

<img src="screenshots/Single_predictions%20at%207.17.49%20PM.png" width="100%">

</td>
</tr>

<tr>
<td width="50%" valign="top">

📦 Batch Prediction

<img src="screenshots/Batch_predictions%20test%202026-09-01%20at%207.21.00%20PM.png" width="100%">

</td>
<td width="50%" valign="top">

📈 Sales Analytics

<img src="screenshots/Analytic_dashboard%207.14.23%20PM.png" width="100%">

</td>
</tr>

<tr>
<td width="50%" valign="top">

📄 Submission Prediction

<img src="screenshots/Analytics_Submission%202026-09-01%20at%207.28.47%20PM.png" width="100%">

</td>
<td width="50%" valign="top">

ℹ️ About Application

<img src="screenshots/About_me%20at%207.30.51%20PM.png" width="100%">

</td>
</tr>
</table>

💻 Technology Stack

<table>
<tr>
<td width="33%" valign="top">

🐍 Programming

Python

Pandas

NumPy

</td>
<td width="33%" valign="top">

🤖 Machine Learning

Scikit-learn

XGBoost

LightGBM

CatBoost

Statsmodels

</td>
<td width="34%" valign="top">

📊 Visualization & App

Matplotlib

Seaborn

Plotly

Streamlit

Git

GitHub

</td>
</tr>
</table>

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

🚀 Installation & Setup

1. Clone the Repository

git clone https://github.com/sgl13/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

2. Create a Virtual Environment

python -m venv .venv

macOS / Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate

3. Upgrade pip

python -m pip install --upgrade pip

4. Install Dependencies

pip install -r requirements.txt

5. Verify Installation

python --version
streamlit --version

6. Launch the Dashboard

streamlit run app.py

Then open:

http://localhost:8501

Stop the application with:

Ctrl + C

📓 Notebook

The main forecasting notebook is:

notebooks/
└── RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb

The notebook contains the data analysis, feature engineering, forecasting workflow, and model development.

📦 Project Deliverables

<table>
<tr>
<td width="50%" valign="top">

✅ Source Code

✅ Jupyter Notebook

✅ EDA

✅ Feature Engineering

✅ Trained Machine Learning Models

</td>
<td width="50%" valign="top">

✅ Interactive Streamlit Dashboard

✅ Batch Prediction

✅ Submission Prediction Output

✅ Final Report

✅ Presentation

✅ GitHub Repository

</td>
</tr>
</table>

📊 Project Evaluation

Criteria

Marks

Business Understanding

10

Data Cleaning

10

EDA

15

Feature Engineering

15

Model Development

20

Model Evaluation

10

Dashboard

10

Business Insights

5

Documentation

3

Presentation

2

Total

100

🎓 Learning Outcomes

By completing this project, the following practical skills are demonstrated:

<table>
<tr>
<td width="50%" valign="top">

End-to-end data science workflow

Retail time-series analysis

Data cleaning and preprocessing

Exploratory data analysis

Forecasting feature engineering

Machine-learning model development

</td>
<td width="50%" valign="top">

Ensemble modeling

Model evaluation

Interactive Streamlit dashboard development

Business insight generation

Git and GitHub project management

Technical project presentation

</td>
</tr>
</table>

👨‍💻 Author

<div align="center">

Shivakumar G L

Retail Sales Foresight & Demand Forecasting

Python • Machine Learning • Time Series • Streamlit • AWS

GitHub: @sgl13

“Where predictions become interactive insights.”

</div>

📌 Project Status

<div align="center">

🟢 COMPLETED

</div>

The completed project includes:

Data Preprocessing → EDA → Feature Engineering → Model Development → Ensemble Forecasting → Model Evaluation → Streamlit Dashboard → Batch Prediction → Submission Output

<div align="center">



</div>