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

The test dataset contains dates covering the 15 days after the last date in the training data.

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

A magnitude 7.8 earthquake occurred in Ecuador on April 16, 2016. Relief efforts increased demand for products such as water and other essential goods for several weeks, making this an important event to consider during analysis.

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

The data preparation stage may include:

Loading and inspecting datasets

Handling missing values

Checking duplicate records

Converting date columns into appropriate formats

Validating data types

Checking outliers and unusual sales values

Merging relevant datasets

Preparing data for time-series modeling

📊 Exploratory Data Analysis

EDA should be used to understand sales behavior and identify important patterns.

Suggested analysis includes:

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

Potential features for forecasting include:

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

Feature engineering should be designed carefully to avoid data leakage, especially when creating lag and rolling features.

🤖 Model Development

The project evaluates multiple forecasting and machine-learning approaches.

Recommended Technologies / Models

Scikit-learn

XGBoost

LightGBM

CatBoost

Statsmodels

Time-series forecasting techniques

Models are compared using an appropriate validation strategy for time-series data.

📈 Model Evaluation

Model performance is evaluated using suitable forecasting metrics:

RMSE

MAE

MAPE

RMSLE

For time-series forecasting, the validation strategy should respect chronological order rather than randomly shuffling observations.

Model Comparison

The notebook evaluates LightGBM, XGBoost, and a weighted LightGBM + XGBoost ensemble. The ensemble weights were selected using validation RMSLE.

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

The completed project should provide actionable insights such as:

Which stores generate the highest sales?

Which product families have the highest demand?

How do promotions affect sales?

Which periods show seasonal demand?

How do holidays affect purchasing behavior?

How do external factors influence sales?

Which stores or products require additional inventory planning?

📊 Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring sales trends, generating predictions, analyzing forecast distributions, and downloading prediction outputs.

Dashboard Capabilities

Sales trends

Store and category performance

Single sales prediction

Batch prediction

Forecast distribution

Product-family analysis

Key model metrics

Final prediction download

🖥️ Dashboard Screenshots

The screenshot filenames below match the files currently present in the screenshots/ folder.

🏠 Dashboard Overview



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
│   └── project scripts
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

3. Activate the environment

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

EDA Report

Feature Engineering Notebook

Trained Model

Dashboard

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

After completing the project, the learner should be able to:

Execute an end-to-end data science project

Work with retail time-series data

Perform data cleaning and EDA

Engineer forecasting features

Build forecasting models

Evaluate model performance

Develop interactive dashboards

Generate business recommendations

Present a data science project using industry best practices

👤 Author

Shivakumar G L

📈 Retail Sales Foresight & Demand Forecasting

“Where many learners collaborate, stronger predictions emerge.”

Where predictions become interactive insights.

Python | Machine Learning | Streamlit

📌 Project Status

Status: ✅ Completed

The project includes data preprocessing, exploratory analysis, feature engineering, machine-learning model development, ensemble forecasting, model evaluation, Streamlit visualization, batch prediction, and submission output generation.