Retail Store Sales Forecasting Using Machine Learning and Time Series Analysis

📌 Project Overview

This project focuses on forecasting retail store sales using Machine Learning and Time Series Analysis.

The project is based on sales data from Corporación Favorita, a large Ecuadorian grocery retailer. The goal is to build a forecasting solution that predicts future sales for different product families across multiple stores.

The solution is intended to support:

Inventory planning

Supply chain optimization

Reduction of stock shortages

Data-driven business decisions

The project includes:

Business Understanding

Data Understanding

Data Cleaning

Exploratory Data Analysis (EDA)

Feature Engineering

Feature Selection

Model Development

Model Evaluation

Business Insights

Interactive Dashboard

🎯 Project Objectives

The main objectives of this project are:

Understand the retail sales forecasting business problem

Clean and preprocess the datasets

Perform Exploratory Data Analysis (EDA)

Engineer meaningful time-series and business features

Build and compare multiple forecasting models

Evaluate model performance

Generate business insights

Develop an interactive dashboard

📊 Dataset

The project contains multiple datasets such as historical sales, product information, store information, holidays, transactions, external indicators, and a prediction dataset.

The objective is to predict sales for thousands of product families sold at Favorita stores located in Ecuador.

Main Features

Feature

Description

store_nbr

Identifies the store where products are sold

family

Identifies the type of product sold

sales

Total sales for a product family at a particular store and date

onpromotion

Number of items in a product family that were being promoted

date

Date of the sales record

Dataset Files

File

Description

train.csv

Historical training data containing store, product family, promotion information and sales

test.csv

Data for the dates for which sales need to be predicted

sample_submission.csv

Sample file showing the required submission format

stores.csv

Store metadata including city, state, type and cluster

oil.csv

Daily oil price information

holidays_events.csv

Holiday and event information

transactions.csv

Transaction-related information

submission.csv

Final generated prediction output

The test dataset contains the 15 days after the last date in the training data.

Dataset Link

Google Drive Dataset

🔍 Exploratory Data Analysis

The EDA includes:

Checking missing values

Checking duplicate records

Checking data types

Cleaning and understanding the datasets

Distribution analysis

Sales trend analysis

Store-level sales analysis

Product-family sales analysis

Promotion vs. sales analysis

Monthly and yearly sales patterns

Day-of-week sales patterns

Holiday and event analysis

Correlation analysis

Analysis of external factors such as oil prices

Analysis of major events affecting demand

🛠️ Data Preprocessing

The project performs the following preprocessing steps:

Loading and inspecting datasets

Handling missing values

Checking duplicate records

Converting date columns into appropriate formats

Validating data types

Cleaning and merging relevant datasets

Encoding categorical variables

Creating date-based features

Creating time-series features

Feature selection

Preparing data for machine learning

Feature engineering is performed carefully to avoid data leakage, especially while creating lag and rolling features.

⚙️ Feature Engineering

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

🔎 Important Business Factors

🎉 Holidays and Events

The holidays_events.csv dataset contains information about holidays and events.

Special attention is given to:

Transferred holidays

Bridge days

Work days

Additional holidays

A transferred holiday is officially moved to another date, while bridge and work days can change normal purchasing and working patterns.

🛢️ Oil Prices

Ecuador's economy is influenced by oil prices, so daily oil price information is considered as an external forecasting indicator.

💰 Pay Days

Public-sector wages are paid every two weeks on the 15th and on the last day of the month. These dates may influence supermarket purchasing behavior.

🌍 Earthquake Effect

A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. Relief efforts increased demand for products such as water and other essential goods for several weeks after the earthquake.

🔄 Project Workflow

The project follows this workflow:

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

🤖 Machine Learning

The target variable for the forecasting problem is:

Sales — Future Retail Sales

Models Used

LightGBM

XGBoost

CatBoost

Statistical Time-Series Techniques

Weighted LightGBM + XGBoost Ensemble

Validation Strategy

The validation strategy respects the chronological order of the data rather than randomly shuffling observations.

This is important because future information should not be used when training a forecasting model.

📈 Model Evaluation

The models are evaluated using:

RMSLE

MAE

RMSE

R² Score

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

LightGBM: 15%

XGBoost: 85%

The ensemble weights were selected using validation RMSLE.

🏆 Best Model

The LightGBM + XGBoost weighted ensemble is the final model used for prediction.

Validation performance:

RMSLE: 0.4127

MAE: 76.2772

RMSE: 262.7155

R²: 0.9554

“Where many learners collaborate, stronger predictions emerge.”

💡 Business Insights

The project provides insights into:

Which stores generate the highest sales

Which product families have the highest demand

How promotions affect sales

Seasonal demand patterns

Sales behavior during holidays and events

The influence of external factors

Stores or products requiring additional inventory planning

📊 Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring sales trends, generating predictions, analyzing forecast distributions, and working with prediction outputs.

Dashboard Features

Sales trends

Store and product-family performance

Single sales prediction

Batch prediction

Forecast distribution

Product-family analysis

Model performance metrics

Prediction output

🖥️ Dashboard Overview



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
├── README.md
├── .gitignore
└── .python-version

Note: Keep your virtual environment such as .venv/ or venv/ local and exclude it from Git using .gitignore.

🚀 How to Run the Project

1. Clone the repository

git clone https://github.com/sgl13/Retail-Sales-Forecasting.git
cd Retail-Sales-Forecasting

2. Create a virtual environment

python -m venv .venv

3. Activate the virtual environment

macOS / Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate

4. Upgrade pip

python -m pip install --upgrade pip

5. Install required libraries

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

📓 Notebooks

The project contains the main forecasting notebook:

1. Retail Store Sales Forecasting

RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb

This notebook contains:

Data understanding

Data cleaning

Exploratory data analysis

Feature engineering

Feature selection

Model development

Model evaluation

Forecasting workflow

📦 Project Deliverables

Source Code

Jupyter Notebook

EDA Report

Feature Engineering Notebook

Trained Machine Learning Models

Interactive Streamlit Dashboard

Batch Prediction

Submission Prediction Output

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

After completing this project, the learner should be able to:

Execute an end-to-end data science project

Work with retail time-series data

Perform data cleaning and EDA

Engineer forecasting features

Build forecasting models

Evaluate model performance

Develop interactive dashboards

Generate business recommendations

Present a data science project using industry best practices

🎯 Project Objective

The objective of this project is to analyze historical retail sales data and build a machine learning forecasting solution that can predict future sales for different stores and product families.

Skills Demonstrated

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

Data Cleaning

Exploratory Data Analysis

Feature Engineering

Feature Selection

Time-Series Analysis

Machine Learning

Forecasting

Ensemble Modeling

Model Evaluation

Streamlit

Git & GitHub

👨‍💻 Author

Shivakumar G L

Retail Sales Foresight & Demand Forecasting

“Where predictions become interactive insights.”

Python | Machine Learning | Time Series | Streamlit

GitHub Profile