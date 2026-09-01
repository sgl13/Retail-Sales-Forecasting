# ==========================================================
# Import Libraries
# ==========================================================

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from predictions import predict_batch, predict_sales
from feature_engineering import engineer_batch_features, MISSING_REFERENCE_FILES

# ==========================================================
# Streamlit Configuration (MUST BE FIRST STREAMLIT CALL)
# ==========================================================

st.set_page_config(
    page_title="Retail Sales Forecasting", page_icon="📈", layout="wide"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    """
<style>
.main {
    background-color: #f7f9fc;
}
h1, h2, h3 {
    color: #0E4D92;
}
div[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,.10);
}
.stButton>button {
    width: 100%;
    background: #1565C0;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 17px;
    font-weight: bold;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# Helper Function for File Resolution & Date Merging
# ==========================================================


def enrich_with_test_metadata(sub_df):
    """If the uploaded batch file is missing 'date' (e.g. sample_submission.csv style with just id/sales),
    merge date/store_nbr/family/onpromotion from test.csv using 'id'.
    Returns (df, merged_bool, error_message_or_None).
    """
    if "date" in sub_df.columns:
        return sub_df, False, None

    if "id" not in sub_df.columns:
        return sub_df, False, (
            "Your file has no 'date' column, and no 'id' column either, "
            "so there's no way to look up dates from test.csv."
        )

    test_paths = [
        "data/sales_retail_timeseries_data/test.csv",
        "data/test.csv",
    ]

    for t_path in test_paths:
        if os.path.exists(t_path):
            test_df = pd.read_csv(t_path)
            cols_to_use = [
                c
                for c in ["id", "date", "store_nbr", "family", "onpromotion"]
                if c in test_df.columns
            ]
            if "date" not in cols_to_use:
                continue

            merged = sub_df.merge(test_df[cols_to_use], on="id", how="left")

            if merged["date"].isna().all():
                return sub_df, False, (
                    f"Found {t_path}, but none of the 'id' values in your "
                    f"upload matched it, so 'date' couldn't be filled in."
                )

            return merged, True, None

    return sub_df, False, (
        "Your file has no 'date' column (sample_submission.csv format). "
        "Please ensure test.csv is placed in the `data/` folder for metadata lookup."
    )


def load_submission_data():
    """Finds submission CSV and automatically merges 'date' (and metadata) from test.csv if missing."""
    submission_paths = [
        "data/submission.csv",
        "data/sales_retail_timeseries_data/sample_submission.csv",
    ]
    test_paths = [
        "data/sales_retail_timeseries_data/test.csv",
        "data/test.csv",
    ]

    sub_df = None
    for path in submission_paths:
        if os.path.exists(path):
            sub_df = pd.read_csv(path)
            break

    if sub_df is None:
        return None

    if "date" not in sub_df.columns and "id" in sub_df.columns:
        for t_path in test_paths:
            if os.path.exists(t_path):
                test_df = pd.read_csv(t_path)
                cols_to_use = [
                    c
                    for c in ["id", "date", "store_nbr", "family"]
                    if c in test_df.columns
                ]
                sub_df = sub_df.merge(test_df[cols_to_use], on="id", how="left")
                break

    return sub_df


# ==========================================================
# Load Metrics
# ==========================================================

try:
    metrics = joblib.load("models/metrics.pkl")
except Exception as e:
    metrics = {}
    st.warning(
        f"Could not load models/metrics.pkl ({e}) - showing placeholder metrics."
    )

# ==========================================================
# Sidebar Navigation
# ==========================================================

st.sidebar.title("📈 Retail Sales Forecasting")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Single Prediction", "Batch Prediction", "Analytics", "About"],
)

# ==========================================================
# Dashboard
# ==========================================================

if page == "Dashboard":

    st.title("Retail Sales Forecasting Dashboard")
    st.write("Predict future retail sales using an Ensemble Machine Learning Model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")

    with col2:
        st.metric("RMSLE", f"{metrics.get('RMSLE', 0):.4f}")

    with col3:
        st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")

    st.markdown("---")

    df = load_submission_data()
    if df is not None and "sales" in df.columns:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            chart_df = df.groupby("date")["sales"].sum().reset_index()
            fig = px.line(
                chart_df,
                x="date",
                y="sales",
                title="Aggregated Sales Forecast Trend Over Time",
            )
        else:
            fig = px.line(
                df.head(200), y="sales", title="Sample Sales Forecast"
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No submission data found - sample chart unavailable.")

    st.info(
        "Use the sidebar to access Single Prediction, Batch Prediction, and Analytics."
    )

# ==========================================================
# Single Prediction
# ==========================================================

elif page == "Single Prediction":

    st.title("🔮 Single Sales Prediction")
    st.write("Enter feature values to predict retail sales for a specific store item.")

    with st.form("prediction_form"):

        st.subheader("Store Information")
        col1, col2 = st.columns(2)

        with col1:
            store_nbr = st.number_input("Store Number", min_value=1, value=1)
            family = st.text_input("Product Family", value="GROCERY I")
            city = st.text_input("City", value="Quito")
            state = st.text_input("State", value="Pichincha")
            store_type = st.text_input("Store Type", value="D")
            cluster = st.number_input("Cluster", min_value=1, value=13)

        with col2:
            onpromotion = st.number_input("On Promotion Count", min_value=0, value=0)
            is_weekend = st.selectbox("Weekend", [0, 1])
            dcoilwtico = st.number_input("Oil Price ($)", value=60.0)
            oil_roll_7 = st.number_input("Oil Rolling 7 Days", value=60.0)
            oil_fwd_1 = st.number_input("Oil Forward 1 Day", value=60.0)
            oil_fwd_3 = st.number_input("Oil Forward 3 Days", value=60.0)

        st.subheader("Calendar & Temporal Features")
        col3, col4 = st.columns(2)

        with col3:
            is_holiday = st.selectbox("Holiday", [0, 1])
            day = st.slider("Day of Month", 1, 31, 15)
            dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

        with col4:
            isweekend = st.selectbox("Is Weekend Flag", [0, 1])
            is_payday = st.selectbox("Payday Flag", [0, 1])

        st.subheader("Historical Sales Dynamics")

        sales_lag_21 = st.number_input("Sales Lag (21 Days Ago)", value=2500.0)
        sales_lag_28 = st.number_input("Sales Lag (28 Days Ago)", value=2600.0)
        sales_roll_21_7 = st.number_input("7-Day Sales Rolling Mean", value=2550.0)
        promo_roll_3 = st.number_input("3-Day Promotion Rolling Mean", value=0.0)

        submit = st.form_submit_button("Predict Sales")

    if submit:
        user_data = {
            "store_nbr": [store_nbr],
            "family": [family],
            "onpromotion": [onpromotion],
            "is_weekend": [is_weekend],
            "city": [city],
            "state": [state],
            "store_type": [store_type],
            "cluster": [cluster],
            "dcoilwtico": [dcoilwtico],
            "oil_roll_7": [oil_roll_7],
            "oil_fwd_1": [oil_fwd_1],
            "oil_fwd_3": [oil_fwd_3],
            "is_holiday": [is_holiday],
            "day": [day],
            "dayofweek": [dayofweek],
            "isweekend": [isweekend],
            "is_payday": [is_payday],
            "sales_lag_21": [sales_lag_21],
            "sales_lag_28": [sales_lag_28],
            "sales_roll_21_7": [sales_roll_21_7],
            "promo_roll_3": [promo_roll_3],
        }

        try:
            input_df = pd.DataFrame(user_data)
            prediction = predict_sales(input_df)

            if isinstance(prediction, (np.ndarray, list, pd.Series)):
                prediction = prediction[0]

            st.success("Prediction Completed Successfully ✅")
            st.metric("Predicted Sales (Units)", f"{max(0.0, float(prediction)):,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================================
# Batch Prediction
# ==========================================================

elif page == "Batch Prediction":

    st.title("📂 Batch Prediction")

    if MISSING_REFERENCE_FILES:
        st.warning(
            "⚠️ Missing reference data: " + ", ".join(MISSING_REFERENCE_FILES) +
            ". Predictions will still run, but missing features will be imputed with 0."
        )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        raw_df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview (Raw Upload)")
        st.dataframe(raw_df.head())

        raw_df, was_merged, merge_error = enrich_with_test_metadata(raw_df)

        if was_merged:
            st.info(
                "Auto-merged missing date/store_nbr/family metadata from test.csv using 'id'."
            )
        elif merge_error and "date" not in raw_df.columns:
            st.warning(merge_error)

        if st.button("Predict Entire Dataset"):
            try:
                # 1. Feature Engineering
                engineered_df = engineer_batch_features(raw_df)

                # 2. Generate Predictions
                result = predict_batch(engineered_df)

                # Ensure 'sales' column exists in result
                if "sales" not in result.columns:
                    if "prediction" in result.columns:
                        result = result.rename(columns={"prediction": "sales"})
                    elif isinstance(result, (pd.Series, np.ndarray, list)):
                        result_df = engineered_df.copy()
                        result_df["sales"] = result
                        result = result_df

                # 3. Explicitly Pin Key Columns (id, sales, date) to the Front
                front_priority = ["id", "sales", "date", "store_nbr", "family"]
                existing_front = [c for c in front_priority if c in result.columns]
                remaining_cols = [c for c in result.columns if c not in existing_front]

                ordered_result = result[existing_front + remaining_cols]

                # Save submission file
                os.makedirs("data", exist_ok=True)
                if 'id' in ordered_result.columns and 'sales' in ordered_result.columns:
                    sub_df = ordered_result[['id', 'sales']].copy()
                    sub_df['id'] = sub_df['id'].astype(int)
                    sub_df.to_csv("data/submission.csv", index=False)
                else:
                    ordered_result.to_csv("data/submission.csv", index=False)

                st.success("Prediction Completed Successfully ✅")

                # 4. Render Features & Prediction Results
                st.subheader("Engineered Features Preview")
                st.dataframe(engineered_df.head())

                st.subheader("Prediction Results Preview")
                st.dataframe(ordered_result.head(10))

                # 5. Download Action Buttons
                col_d1, col_d2 = st.columns(2)

                with col_d1:
                    full_csv = ordered_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Predictions CSV",
                        data=full_csv,
                        file_name="full_predictions.csv",
                        mime="text/csv",
                    )

                with col_d2:
                    if 'id' in ordered_result.columns and 'sales' in ordered_result.columns:
                        submission_df = ordered_result[['id', 'sales']].copy()
                        submission_df['id'] = submission_df['id'].astype(int)
                        submission_csv = submission_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Kaggle Submission CSV",
                            data=submission_csv,
                            file_name="submission.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ==========================================================
# Analytics
# ==========================================================

elif page == "Analytics":

    st.title("📊 Analytics & Sales Trend")

    df = load_submission_data()

    if df is not None and "sales" in df.columns:

        st.subheader("💡 Sales Summary Overview")
        m1, m2, m3, m4 = st.columns(4)

        total_sales = df["sales"].sum()
        avg_sales = df["sales"].mean()
        max_sales = df["sales"].max()
        records_count = len(df)

        with m1:
            st.metric("Total Forecasted Sales", f"{total_sales:,.0f}")
        with m2:
            st.metric("Average Sales / Row", f"{avg_sales:,.2f}")
        with m3:
            st.metric("Peak Prediction", f"{max_sales:,.2f}")
        with m4:
            st.metric("Total Rows Analyzed", f"{records_count:,}")

        st.markdown("---")

        st.subheader("📈 Overall Sales Trend Over Time")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            daily_df = df.groupby("date")["sales"].sum().reset_index()

            fig_trend = px.line(
                daily_df,
                x="date",
                y="sales",
                title="Total Daily Sales Aggregated Across All Stores",
                labels={"date": "Date", "sales": "Total Sales Units"},
                template="plotly_white",
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            df["moving_avg"] = df["sales"].rolling(window=50, min_periods=1).mean()
            fig_trend = px.line(
                df.head(500),
                y=["sales", "moving_avg"],
                title="Sales Predictions with 50-Row Moving Average",
                labels={"value": "Sales Units", "index": "Prediction Row"},
                template="plotly_white",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📊 Sales Distribution")
            fig_hist = px.histogram(
                df,
                x="sales",
                nbins=40,
                title="Distribution of Forecast Values",
                color_discrete_sequence=["#1565C0"],
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.subheader("🏪 Category / Store Performance")
            if "family" in df.columns:
                fam_sales = (
                    df.groupby("family")["sales"]
                    .sum()
                    .reset_index()
                    .sort_values(by="sales", ascending=False)
                )
                fig_bar = px.bar(
                    fam_sales.head(10),
                    x="sales",
                    y="family",
                    orientation="h",
                    title="Top Product Families by Sales",
                    color="sales",
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            elif "store_nbr" in df.columns:
                store_sales = (
                    df.groupby("store_nbr")["sales"]
                    .sum()
                    .reset_index()
                    .sort_values(by="sales", ascending=False)
                )
                fig_bar = px.bar(
                    store_sales.head(10),
                    x="store_nbr",
                    y="sales",
                    title="Top Stores by Forecasted Sales",
                    color="sales",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning(
            "No submission/sales data found - run Batch Prediction or add data/submission.csv."
        )

# ==========================================================
# About
# ==========================================================

elif page == "About":

    st.title("ℹ️ About")

    st.markdown(
        """
## Retail Sales Forecasting

This interactive dashboard predicts future retail sales using an ensemble LightGBM and XGBoost model.

### Features
- Single Prediction
- Batch Prediction
- Interactive Analytics & Time-Series Visualizations
- Download Predictions

### Author
**Shivakumar G L**  
*Where predictions become interactive insights*
"""
    )