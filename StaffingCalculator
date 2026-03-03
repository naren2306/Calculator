# WFM Capacity & Forecasting Studio (Streamlit Version)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- SYNTHETIC DATA ----------------

def generate_synthetic_data():

    np.random.seed(42)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=180)

    volume = (
        1200
        + 150 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        + 80 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
        + np.random.normal(0, 60, len(dates))
    )

    df = pd.DataFrame({
        "date": dates,
        "volume": np.round(volume)
    })

    return df


# ---------------- METRICS ----------------

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


# ---------------- UI ----------------

st.set_page_config(page_title="WFM Capacity & Forecasting Studio", layout="wide")

st.title("WFM Capacity & Forecasting Studio")


tabs = st.tabs(["FTE Calculator", "Forecasting Playground"])


# ======================================================
# FTE CALCULATOR
# ======================================================

with tabs[0]:

    st.header("FTE Calculator")

    col1, col2 = st.columns([1,2])

    with col1:

        volume = st.number_input("Daily Contact Volume", value=1000)
        aht = st.number_input("Average Handle Time (seconds)", value=300)
        occupancy = st.number_input("Target Occupancy (%)", value=85)
        shrinkage = st.number_input("Shrinkage (%)", value=30)

    workload_hours = (volume * aht) / 3600
    base_fte = workload_hours / 8
    occ_adj = base_fte / (occupancy / 100)
    final_fte = occ_adj / (1 - shrinkage / 100)

    with col2:

        st.subheader("Required FTE")

        st.metric("FTE Required", round(final_fte,2))

        fte_df = pd.DataFrame({
            "Metric":["Base FTE","Occupancy Adjusted","Final Required FTE"],
            "Value":[base_fte,occ_adj,final_fte]
        })

        fig, ax = plt.subplots()

        ax.bar(fte_df["Metric"], fte_df["Value"])

        ax.set_ylabel("FTE")
        ax.set_title("FTE Breakdown")

        st.pyplot(fig)


# ======================================================
# FORECASTING PLAYGROUND
# ======================================================

with tabs[1]:

    st.header("Forecasting Playground")

    col1, col2 = st.columns([1,3])

    # ---------------- CONTROLS ----------------

    with col1:

        uploaded_file = st.file_uploader("Upload CSV (date,volume)")

        horizon = st.number_input("Forecast Horizon (Days)", value=14)

        model_choice = st.selectbox(
            "Select Forecast Model",
            [
                "Best Model",
                "ARIMA",
                "ETS",
                "Seasonal Naive",
                "Prophet",
                "TimeGPT (Placeholder)"
            ]
        )

        synthetic = generate_synthetic_data()

        st.download_button(
            "Download Synthetic Dataset",
            synthetic.to_csv(index=False),
            file_name="synthetic_wfm_volume_data.csv"
        )


    # ---------------- DATA INPUT ----------------

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

    else:

        df = synthetic.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    ts = df["volume"]


    # ---------------- TRAIN / TEST SPLIT ----------------

    train_size = int(len(ts)*0.8)

    train = ts[:train_size]
    test = ts[train_size:]


    # ---------------- MODELS ----------------

    # ARIMA
    arima_model = ARIMA(train, order=(2,1,2)).fit()
    arima_pred = arima_model.forecast(len(test))


    # ETS
    ets_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=7).fit()
    ets_pred = ets_model.forecast(len(test))


    # Seasonal Naive
    snaive_pred = train.shift(7).dropna()
    snaive_pred = snaive_pred[-len(test):]


    # Prophet
    prophet_df = df.iloc[:train_size].rename(columns={"date":"ds","volume":"y"})

    prophet_model = Prophet()

    prophet_model.fit(prophet_df)

    future = prophet_model.make_future_dataframe(periods=len(test))

    prophet_fc = prophet_model.predict(future)

    prophet_pred = prophet_fc["yhat"].tail(len(test))


    # TimeGPT placeholder (using ARIMA)
    timegpt_pred = arima_pred


    # ---------------- ACCURACY TABLE ----------------

    accuracy = pd.DataFrame({

        "Model":[
            "ARIMA",
            "ETS",
            "Seasonal Naive",
            "Prophet",
            "TimeGPT"
        ],

        "MAE":[
            mean_absolute_error(test,arima_pred),
            mean_absolute_error(test,ets_pred),
            mean_absolute_error(test,snaive_pred),
            mean_absolute_error(test,prophet_pred),
            mean_absolute_error(test,timegpt_pred)
        ],

        "RMSE":[
            np.sqrt(mean_squared_error(test,arima_pred)),
            np.sqrt(mean_squared_error(test,ets_pred)),
            np.sqrt(mean_squared_error(test,snaive_pred)),
            np.sqrt(mean_squared_error(test,prophet_pred)),
            np.sqrt(mean_squared_error(test,timegpt_pred))
        ],

        "MAPE":[
            mape(test,arima_pred),
            mape(test,ets_pred),
            mape(test,snaive_pred),
            mape(test,prophet_pred),
            mape(test,timegpt_pred)
        ]

    })


    accuracy = accuracy.sort_values("RMSE")


    best_model = accuracy.iloc[0]["Model"]


    if model_choice == "Best Model":

        selected_model = best_model

    else:

        selected_model = model_choice


    # ---------------- FINAL FORECAST ----------------

    if selected_model == "ARIMA":

        forecast = arima_model.forecast(horizon)

    elif selected_model == "ETS":

        forecast = ets_model.forecast(horizon)

    elif selected_model == "Seasonal Naive":

        forecast = ts.shift(7).dropna().iloc[-horizon:]

    elif selected_model == "Prophet":

        prophet_df = df.rename(columns={"date":"ds","volume":"y"})

        model = Prophet()

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=horizon)

        fc = model.predict(future)

        forecast = fc["yhat"].tail(horizon)

    else:

        forecast = arima_model.forecast(horizon)


    future_dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=horizon
    )


    forecast_df = pd.DataFrame({
        "date":future_dates,
        "volume":forecast
    })


    # ---------------- PLOT ----------------

    with col2:

        fig, ax = plt.subplots()

        ax.plot(df["date"], df["volume"], label="Actual")

        ax.plot(forecast_df["date"], forecast_df["volume"], label="Forecast")

        ax.set_title(f"Forecast using: {selected_model}")

        ax.legend()

        st.pyplot(fig)

        st.subheader("Model Accuracy Comparison")

        st.dataframe(accuracy.round(3))
