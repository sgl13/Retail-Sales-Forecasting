README.md


🛒 Retail Store Sales Forecasting
Machine Learning + Time Series Analysis + Streamlit Dashboard

📌 Project Overview
Project	Details
Domain	Retail Analytics
Company Dataset	Corporación Favorita
Problem	Future Retail Sales Forecasting
Target	Sales
Approach	Machine Learning + Time Series Analysis
Final Model	LightGBM + XGBoost Ensemble
Application	Interactive Streamlit Dashboard
Status	✅ Completed
Project Workflow
Data Collection → Data Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Ensemble Model → Prediction → Dashboard

Project Includes


🧹 Data Cleaning	📊 Exploratory Data Analysis
⚙️ Feature Engineering	🎯 Feature Selection
⏱️ Time-Series Features	🤖 Machine Learning
📈 Model Evaluation	🏆 Ensemble Modeling
💡 Business Insights	🖥️ Streamlit Dashboard
📊 Dataset
The project uses multiple datasets containing historical sales, store information, product families, holidays, transactions, oil prices, and prediction data.

Dataset Files
File	Description
train.csv	Historical training data
test.csv	Data used for future sales prediction
sample_submission.csv	Required submission format
stores.csv	Store information
oil.csv	Daily oil price information
holidays_events.csv	Holiday and event information
transactions.csv	Transaction information
submission.csv	Final prediction output
Main Features
Feature	Description	Feature	Description
store_nbr	Store identifier	family	Product family
sales	Total sales	onpromotion	Products on promotion
date	Sales date	oil	External oil-price indicator
The test dataset contains the 15 days after the last date in the training data.

Dataset Link
Google Drive Dataset

🔍 Exploratory Data Analysis
Data Analysis	Business Analysis
Missing-value analysis	Store-wise sales
Duplicate-value analysis	Product-family sales
Data-type checking	Promotion vs. sales
Sales distribution	Monthly sales trends
Correlation analysis	Yearly sales trends
Distribution analysis	Day-of-week sales
Holiday analysis	External-factor analysis
Major-event analysis	Oil-price analysis
🛠️ Data Preprocessing
Data Preparation	Feature Preparation
Handle missing values	Create date features
Check duplicate records	Create time-series features
Convert date columns	Encode categorical variables
Validate data types	Feature selection
Merge relevant datasets	Prepare model-ready data
Clean inconsistent data	Avoid data leakage
Time-series features are created carefully so that future information is not used during model training.

⚙️ Feature Engineering
📅 Date Features
Feature	Purpose	Feature	Purpose
Year	Yearly trend	Month	Monthly seasonality
Day	Daily pattern	Day of Week	Weekly pattern
Week of Year	Weekly seasonality	Quarter	Quarterly trend
Weekend	Weekend demand	—	—
⏱️ Time-Series Features
Feature	Feature	Feature	Feature
Lag Sales	Rolling Average	Rolling Sum	Previous-Day Sales
Previous-Week Sales	Previous-Month Sales	Historical Sales	Demand Patterns
🏪 Business Features
Feature	Feature	Feature	Feature
Store Number	Product Family	Promotion	Holiday
Store Cluster	Store Type	Oil Price	Store-Level Information
🔎 Important Business Factors
Factor	Description
🎉 Holidays & Events	Can influence customer demand
🛢️ Oil Prices	External economic indicator
💰 Pay Days	May influence purchasing behavior
🌍 Earthquake	Increased demand for essential products
Holidays and Events
The holidays_events.csv dataset contains information about holidays and events, including transferred holidays, bridge days, work days, and additional holidays.

Oil Prices
Daily oil price information is used as an external forecasting indicator.

Pay Days
Public-sector wages are paid every two weeks on the 15th and last day of the month.

Earthquake Effect
A magnitude 7.8 earthquake occurred in Ecuador on April 16, 2016, increasing demand for essential products during relief efforts.

🤖 Machine Learning
Target Variable
Sales — Future Retail Sales

Models Used
Machine Learning Models	Forecasting Approach
LightGBM	Time-Series Techniques
XGBoost	Lag-Based Features
CatBoost	Rolling Features
LightGBM + XGBoost	Ensemble Forecasting
Validation Strategy
The project uses chronological validation instead of randomly shuffling time-series data.

📈 Model Evaluation
Evaluation Metrics
Metric	Purpose
RMSLE	Measures logarithmic prediction error
MAE	Measures average absolute error
RMSE	Measures prediction error magnitude
R² Score	Measures explained variance
Model Comparison
Model	Metric	Score
LightGBM	Validation L2 (MSE)	0.1861
XGBoost	Individual metric not recorded	—
LightGBM + XGBoost Ensemble	RMSLE	0.4127
LightGBM + XGBoost Ensemble	MAE	76.2772
LightGBM + XGBoost Ensemble	RMSE	262.7155
LightGBM + XGBoost Ensemble	R²	0.9554
🏆 Final Ensemble
Model	Weight
LightGBM	15%
XGBoost	85%
Final Validation Performance
Metric	Result
RMSLE	0.4127
MAE	76.2772
RMSE	262.7155
R²	0.9554
Final Model: Weighted LightGBM + XGBoost Ensemble

💡 Business Insights
Sales Insights	Planning Insights
Highest-performing stores	Inventory planning
Highest-demand product families	Supply chain optimization
Promotion impact on sales	Store-level planning
Seasonal demand patterns	Product-level planning
Holiday demand changes	Demand forecasting
External-factor impact	Data-driven decisions
📊 Interactive Streamlit Dashboard
Dashboard Modules
Module	Purpose	Module	Purpose
🏠 Dashboard Overview	Application overview	🔮 Single Prediction	Individual sales prediction
📦 Batch Prediction	Multiple sales predictions	📈 Sales Analytics	Sales and forecast analysis
📄 Submission Prediction	Prediction output	ℹ️ About Application	Project information
🖼️ Dashboard Screenshots
<table> <tr> <td width="50%" align="center">

🏠 Dashboard Overview
<img src="./screenshots/Dashboard_Preview%20at%207.10.55%20PM.png" width="100%">

</td> <td width="50%" align="center">

🔮 Single Prediction
<img src="./screenshots/Single_predictions%20at%207.17.49%20PM.png" width="100%">

</td> </tr>

<tr> <td width="50%" align="center">

📦 Batch Prediction
<img src="./screenshots/Batch_predictions%20test%202026-09-01%20at%207.21.00%20PM.png" width="100%">

</td> <td width="50%" align="center">

📈 Sales Analytics
<img src="./screenshots/Analytic_dashboard%207.14.23%20PM.png" width="100%">

</td> </tr>

<tr> <td width="50%" align="center">

📄 Submission Prediction
<img src="./screenshots/Analytics_Submission%202026-09-01%20at%207.28.47%20PM.png" width="100%">

</td> <td width="50%" align="center">

ℹ️ About Application
<img src="./screenshots/About_me%20at%207.30.51%20PM.png" width="100%">

</td> </tr> </table>

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
Operating System	Command
macOS / Linux	source .venv/bin/activate
Windows	.venv\Scripts\activate
4. Install Required Libraries
pip install -r requirements.txt
5. Run the Streamlit Application
streamlit run app.py
6. Open the Application
http://localhost:8501
7. Stop the Application
Ctrl + C
📓 Notebook
Main Notebook
RetailStoreSales_Forecasting_Using_Machine_Learning_&_Time_Series_Analysis.ipynb
Notebook Includes
Analysis	Modeling
Data Understanding	Model Training
Data Cleaning	Model Evaluation
Exploratory Data Analysis	Ensemble Modeling
Feature Engineering	Forecasting
Feature Selection	Prediction
📦 Project Deliverables
Deliverable	Status	Deliverable	Status
Source Code	✅	Jupyter Notebook	✅
EDA Report	✅	Feature Engineering	✅
Trained Models	✅	Streamlit Dashboard	✅
Batch Prediction	✅	Submission Prediction	✅
Final Report	✅	Presentation	✅
GitHub Repository	✅	Demonstration Video	✅
📝 Project Evaluation
Criteria	Marks	Criteria	Marks
Business Understanding	10	Data Cleaning	10
EDA	15	Feature Engineering	15
Model Development	20	Model Evaluation	10
Dashboard	10	Business Insights	5
Documentation	3	Presentation	2
Total	100	

🎓 Learning Outcomes
Technical Skills	Project Skills
Python Programming	End-to-End Data Science
Pandas & NumPy	Data Cleaning
Matplotlib & Seaborn	Exploratory Data Analysis
Scikit-learn	Feature Engineering
XGBoost	Time-Series Analysis
LightGBM	Ensemble Modeling
CatBoost	Model Evaluation
Plotly & Streamlit	Business Analytics
Git & GitHub	Project Presentation
🎯 Project Objective
Objective	Business Application
Forecast future sales	Inventory Planning
Understand demand patterns	Supply Chain Optimization
Analyze store performance	Store-Level Planning
Analyze product demand	Product-Level Planning
Identify sales drivers	Data-Driven Decisions
👨‍💻 Author
Shivakumar G L
Retail Sales Foresight & Demand Forecasting

“Where predictions become interactive insights.”

Technology	Focus
Python	Data & Machine Learning
Machine Learning	Sales Forecasting
Time Series	Demand Analysis
Streamlit	Interactive Dashboard
Git & GitHub	Version Control
GitHub Profile

📌 Project Status
Completed ✅

Component	Status
Data Preprocessing	✅
EDA	✅
Feature Engineering	✅
Machine Learning	✅
Ensemble Forecasting	✅
Model Evaluation	✅
Streamlit Dashboard	✅
Batch Prediction	✅
Submission Output	✅