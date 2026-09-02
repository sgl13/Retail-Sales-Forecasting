🛒 Retail Store Sales Forecasting Using Machine Learning and Time Series Analysis

📌 Project Overview

This project focuses on analyzing retail sales data and building a machine learning model to forecast future sales.

The dataset is based on Corporación Favorita, a large Ecuadorian grocery retailer.

Project Includes

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Engineering

Feature Selection

Time-Series Feature Creation

Machine Learning Model Training

Model Evaluation

Ensemble Modeling

Business Insights

Streamlit Dashboard

📊 Dataset

The project uses multiple datasets containing historical sales, store information, product families, holidays, transactions, oil prices, and prediction data.

Main Dataset Files

File

Description

train.csv

Historical training data

test.csv

Data used for future sales prediction

sample_submission.csv

Required submission format

stores.csv

Store information

oil.csv

Daily oil price information

holidays_events.csv

Holiday and event information

transactions.csv

Transaction information

submission.csv

Final prediction output

Main Features

Feature

Description

store_nbr

Store identifier

family

Product family

sales

Total sales

onpromotion

Number of products on promotion

date

Sales date

The test dataset contains the 15 days after the last date in the training data.

Dataset Link

Google Drive Dataset

🔍 Exploratory Data Analysis

Analysis Performed

Missing-value analysis

Duplicate-value analysis

Data-type checking

Sales distribution analysis

Store-wise sales analysis

Product-family analysis

Promotion vs. sales analysis

Monthly sales trends

Yearly sales trends

Day-of-week analysis

Holiday and event analysis

Correlation analysis

Oil-price analysis

Major-event analysis

🛠️ Data Preprocessing

Preprocessing Steps

Load and inspect datasets

Handle missing values

Check duplicate records

Convert date columns

Validate data types

Merge relevant datasets

Encode categorical variables

Create date-based features

Create time-series features

Select relevant features

Prepare data for model training

Time-series features are created carefully to avoid data leakage.

⚙️ Feature Engineering

📅 Date Features

Feature

Purpose

Year

Yearly trend

Month

Monthly seasonality

Day

Daily pattern

Day of Week

Weekly pattern

Week of Year

Weekly seasonality

Quarter

Quarterly trend

Weekend

Weekend demand

⏱️ Time-Series Features

Lag Sales

Rolling Average

Rolling Sum

Previous-Day Sales

Previous-Week Sales

Previous-Month Sales

🏪 Business Features

Store Number

Product Family

Promotion

Holiday

Store Cluster

Store Type

Oil Price

🔎 Important Business Factors

Factor

Description

🎉 Holidays & Events

Can influence customer demand

🛢️ Oil Prices

External economic indicator

💰 Pay Days

May influence purchasing behavior

🌍 Earthquake

Increased demand for essential products

Holidays and Events

The holidays_events.csv dataset contains information about holidays and events, including transferred holidays, bridge days, work days, and additional holidays.

Oil Prices

Daily oil price information is used as an external forecasting indicator.

Pay Days

Public-sector wages are paid every two weeks on the 15th and last day of the month.

Earthquake Effect

A magnitude 7.8 earthquake occurred in Ecuador on April 16, 2016, which increased demand for essential products during relief efforts.

🤖 Machine Learning

Target Variable

Sales — Future Retail Sales

Models Used

LightGBM

XGBoost

CatBoost

Statistical Time-Series Techniques

Weighted LightGBM + XGBoost Ensemble

Validation Strategy

The project uses chronological validation instead of randomly shuffling the time-series data.

📈 Model Evaluation

Evaluation Metrics

RMSLE

MAE

RMSE

R² Score

Model Comparison

Model

Metric

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

🏆 Final Model

Weighted LightGBM + XGBoost Ensemble

Model

Weight

LightGBM

15%

XGBoost

85%

Final Validation Performance

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

“Where many learners collaborate, stronger predictions emerge.”

💡 Business Insights

The Project Helps Identify

Highest-performing stores

Highest-demand product families

Promotion impact on sales

Seasonal demand patterns

Holiday demand changes

External-factor impact

Inventory planning opportunities

📊 Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for sales analysis and prediction.

Dashboard Modules

Module

Purpose

🏠 Dashboard Overview

Application overview

🔮 Single Prediction

Individual sales prediction

📦 Batch Prediction

Multiple sales predictions

📈 Sales Analytics

Sales and forecast analysis

📄 Submission Prediction

Prediction output

ℹ️ About Application

Project information

🖼️ Dashboard Screenshots

Dashboard Overview



Single Prediction



Batch Prediction



Sales Analytics



Submission Prediction



About Application



📁 Project Structure

Retail-Sales-Forecasting/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   ├── transactions.csv
│   ├── sample_submission.csv
│   └── submission.csv
│
├── models/
│   ├── ensemble_weight.pkl
│   ├── features_columns.pkl
│   ├── label_encoders.pkl
│   ├── lightgbm_model.pkl
│   ├── xgboost_model.pkl
│   └── metrics.pkl
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
│   ├── Dashboard_Preview at 7.10.55 PM.png
│   ├── Single_predictions at 7.17.49 PM.png
│   ├── Batch_predictions test 2026-09-01 at 7.21.00 PM.png
│   ├── Analytic_dashboard 7.14.23 PM.png
│   ├── Analytics_Submission 2026-09-01 at 7.28.47 PM.png
│   └── About_me at 7.30.51 PM.png
│
├── app.py
├── main.py
├── predictions.py
├── requirements.txt
├── .gitignore
└── README.md

🚀 How to Run the Project

1. Clone the Repository

git clone https://github.com/sgl13/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

2. Create a Virtual Environment

python -m venv .venv

3. Activate the Environment

macOS / Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate

4. Install Required Libraries

pip install -r requirements.txt

5. Run the Streamlit Application

streamlit run app.py

Open the local URL displayed in the Terminal.

http://localhost:8501

6. Stop the Application

Ctrl + C

📓 Notebook

Main Notebook

RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb

Notebook Includes

Data Understanding

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Feature Selection

Model Training

Model Evaluation

Forecasting

📦 Project Deliverables

Source Code

Jupyter Notebook

EDA Report

Feature Engineering

Trained Models

Streamlit Dashboard

Batch Prediction

Submission Prediction

Final Report (PDF)

Presentation

GitHub Repository

Project Demonstration Video

📝 Project Evaluation

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

Skills Developed

Python Programming

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Time-Series Analysis

Machine Learning

Ensemble Modeling

Model Evaluation

Streamlit Dashboard Development

Business Analytics

Git & GitHub

🎯 Project Objective

The objective of this project is to analyze historical retail sales data and build a machine learning forecasting solution that predicts future sales for different stores and product families.

Business Applications

Inventory Planning

Supply Chain Optimization

Store-Level Planning

Product-Level Planning

Demand Forecasting

Data-Driven Business Decisions

👨‍💻 Author

Shivakumar G L

Retail Sales Foresight & Demand Forecasting

“Where predictions become interactive insights.”

Python | Machine Learning | Time Series | Streamlit

GitHub Profile

📌 Project Status

Completed ✅

The project includes data preprocessing, EDA, feature engineering, machine learning model development, ensemble forecasting, model evaluation, Streamlit visualization, batch prediction, and submission output generation.