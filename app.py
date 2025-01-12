import os

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from fpdf import FPDF
import tempfile

# Імпорт функцій із ваших модулів
from models.arima.sarima import train_sarima_model
from models.arima.arima import train_arima_model
from models.arima.utils import predict_date_by_date
from models.gbr.prediction import predict_gradient_boosting
from models.gbr.train import train_gradient_boosting
from models.gru.prediction import predict_gru
from models.gru.train import train_gru
from models.lstm.prediction import predict_lstm
from models.lstm.train import train_lstm
from models.prophet.prediction import predict_prophet
from models.prophet.train import train_prophet
from models.random_forest.prediction import predict_random_forest
from models.random_forest.train import train_random_forest
from models.utils import create_lag_features, create_sequences, get_tickers

import warnings
warnings.filterwarnings('ignore')


def load_data(ticker, start_date):
    data = yf.download(ticker, start=start_date).reset_index()
    return data[['Close', 'Date']].rename(columns={'Close': 'rate', 'Date': 'date'})


def display_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mape, r2


def generate_pdf_report(file_path, data_summary, model_name, metrics, forecast_table, fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Заголовок
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Forecasting report", ln=True, align='C')

    # Опис вхідних даних
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Review of the input data", ln=True)
    for key, value in data_summary.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Обрана модель
    pdf.cell(200, 10, txt="", ln=True)  # Пустий рядок
    pdf.cell(200, 10, txt="Chosen model", ln=True)
    pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True)

    # Метрики продуктивності
    pdf.cell(200, 10, txt="", ln=True)  # Пустий рядок
    pdf.cell(200, 10, txt="Metrics", ln=True)
    for metric, value in metrics.items():
        pdf.cell(200, 10, txt=f"{metric}: {value:.2f}", ln=True)

    # Таблиця прогнозів
    pdf.cell(200, 10, txt="", ln=True)  # Пустий рядок
    pdf.cell(200, 10, txt="Results of forecast of the first 25 steps", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(50, 10, txt="Actual", border=1)
    pdf.cell(50, 10, txt="Predicted", border=1, ln=True)
    for row in forecast_table[:25]:
        actual_value = float(row['actual'])
        forecast_value = float(row['forecast'])
        pdf.cell(50, 10, txt=f"{actual_value:.2f}", border=1)
        pdf.cell(50, 10, txt=f"{forecast_value:.2f}", border=1, ln=True)

    # Додавання графіків
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Visualisation", ln=True, align='C')

    # Збереження графіка у тимчасовий файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        fig.write_image(temp_file.name)  # Збереження графіка Plotly у файл
        pdf.image(temp_file.name, x=10, y=30, w=180)  # Додавання графіка до PDF

    # Генерація PDF
    pdf.output(file_path)


def serve_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        return pdf_file.read()



# Streamlit додаток
st.set_page_config(
    page_title="Time series forecasting",
    page_icon="📈"
)

st.title("Time series forecasting")
tickers = get_tickers()

# Завантаження даних
st.sidebar.header("📂 Data settings")
ticker = st.sidebar.selectbox(
        "Stock ticker",
        tickers,
        index=tickers.index('AAPL')
    )
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))

if "data" not in st.session_state:
    st.session_state["data"] = None

if st.sidebar.button("Download Data"):
    st.session_state["data"] = load_data(ticker, start_date)

if st.session_state["data"] is not None:
    data = st.session_state["data"]
    st.header("Downloaded data:")
    st.dataframe(yf.download(ticker, start=start_date).reset_index())
    st.subheader("Visualisation of downloaded data:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['rate'].values.flatten(), mode='lines'))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(fig)

    # Розділення даних
    train_size = int(len(data) * 0.95)
    df_train = data.iloc[:train_size]
    df_test = data.iloc[train_size:]

    st.header("Splitting into test and training data:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['date'], y=df_train['rate'].values.flatten(), mode='lines', name='Training'))
    fig.add_trace(go.Scatter(x=df_test['date'], y=df_test['rate'].values.flatten(), mode='lines', name='Testing'))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white"
    )

    st.plotly_chart(fig)
    # Лагові ознаки
    df_train_ml = create_lag_features(df_train.copy())
    df_train_ml.dropna(inplace=True)
    X_train_ml = df_train_ml[[f'lag_{i}' for i in range(1, 4)]]
    y_train_ml = df_train_ml['rate']
    # Вибираємо ознаки (наприклад, Open, High, Low, Volume) і цільову змінну (Close)
    X = yf.download(ticker, start=start_date).reset_index()[['Open', 'High', 'Low', 'Volume']]
    y = yf.download(ticker, start=start_date).reset_index()['Close']

    # Розділення даних на тренувальний і тестовий набори
    train_size = int(len(data) * 0.95)
    X_train = X.iloc[:train_size][3:]
    X_test = X.iloc[train_size:][3:]
    y_train = y.iloc[:train_size][3:]
    y_test = y.iloc[train_size:][3:]
    y_test_ml = df_test['rate'][3:]

    # Вибір моделі
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "Random Forest"

    st.sidebar.header("🔧 Model Settings")
    model_choice = st.sidebar.selectbox(
        "Model",
        ["Random Forest", "Prophet", "Gradient Boosting", "LSTM", "GRU", "ARIMA", "SARIMA"],
        index=["Random Forest", "Prophet", "Gradient Boosting",
               "LSTM", "GRU", "ARIMA", "SARIMA"].index(st.session_state["model_choice"])
    )
    st.session_state["model_choice"] = model_choice

    if "results" not in st.session_state:
        st.session_state["results"] = None

    if st.sidebar.button("Run the model"):
        if model_choice == "ARIMA":
            model = train_arima_model(df_train)
            forecast = predict_date_by_date(model, y_test_ml, ticker)
        if model_choice == "SARIMA":
            model = train_sarima_model(df_train)
            forecast = predict_date_by_date(model, y_test_ml, ticker)
        if model_choice == "Random Forest":
            rf_model = train_gradient_boosting(X_train, y_train)
            forecast = predict_random_forest(rf_model, X_test)
        elif model_choice == "Gradient Boosting":
            gb_model = train_gradient_boosting(X_train, y_train)
            forecast = predict_gradient_boosting(gb_model, X_test)
        elif model_choice == "Prophet":
            model = train_prophet(df_train)
            forecast = predict_prophet(model, df_test, y_test_ml)
        elif model_choice == "LSTM":
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(df_train[['rate']])
            test_scaled = scaler.transform(df_test[['rate']])
            X_train_dl, y_train_dl = create_sequences(train_scaled, 3)
            X_test_dl, y_test_dl = create_sequences(test_scaled, 3)
            X_train_dl = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], 1)
            X_test_dl = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], 1)
            model = train_lstm(X_train_dl, y_train_dl, 3)
            forecast = predict_lstm(model, X_test_dl, scaler)
        elif model_choice == "GRU":
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(df_train[['rate']])
            test_scaled = scaler.transform(df_test[['rate']])
            X_train_dl, y_train_dl = create_sequences(train_scaled, 3)
            X_test_dl, y_test_dl = create_sequences(test_scaled, 3)
            X_train_dl = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], 1)
            X_test_dl = X_test_dl.reshape(X_test_dl.shape[0], X_test_dl.shape[1], 1)
            model = train_gru(X_train_dl, y_train_dl, 3)
            forecast = predict_gru(model, X_test_dl, scaler)


        # Збереження результатів
        st.session_state["results"] = (y_test_ml, forecast)
    if st.session_state["results"]:
        y_test_ml, forecast = st.session_state["results"]
        rmse, mape, r2 = display_metrics(y_test_ml, forecast)
        # Інтерактивний графік з Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_test['date'][:len(y_test_ml[ticker].values)], y=y_test_ml[ticker].values,
                                 mode='lines', name='Actual data'))
        fig.add_trace(go.Scatter(x=df_test['date'][:len(y_test_ml[ticker].values)], y=forecast, mode='lines',
                                 name='Predicted data'))
        st.header('Prediction results block')
        st.subheader('Prediction and test data graph')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            legend_title="Legend",
            template="plotly_white"
        )

        st.plotly_chart(fig)

        st.subheader('Model metrics')
        metrics = {"RMSE": rmse, "MAPE": mape, "R2": r2}
        st.write(f"**RMSE: {rmse:.2f}**")
        st.write(f"MAPE: {mape:.2%}")
        st.write(f"R-квадрат: {r2:.2f}")

        st.subheader('Table of predicted and actual data')
        df_show = pd.DataFrame()
        df_show['date'] = df_test['date'][:len(y_test_ml[ticker].values)]
        df_show['actual'] = y_test_ml[ticker].values
        df_show['predicted'] = forecast
        st.dataframe(df_show)
        # Дані для PDF
        data_summary = {
            "Training Records": len(df_train),
            "Testing Records": len(df_test)
        }
        forecast_table = [{"actual": float(act), "forecast": float(fore)} for act, fore in zip(y_test_ml.values, forecast)]
        # Завантаження звіту
        st.sidebar.header("Creating a report")
        if st.sidebar.button("Create a report"):
            generate_pdf_report('forecasting_report.pdf', data_summary, model_choice, metrics, forecast_table, fig)
            pdf_data = serve_pdf('forecasting_report.pdf')
            st.sidebar.download_button(
                label="Download PDF report",
                data=pdf_data,
                file_name=f'{ticker}_{model_choice}_forecast_report.pdf',
                mime='application/pdf'
            )
            if os.path.exists(f'forecasting_report.pdf'):
                os.remove(f"forecasting_report.pdf")
