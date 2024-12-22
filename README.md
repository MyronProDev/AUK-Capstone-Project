# Time Series Forecasting Application

## Description
This repository contains a time series forecasting web application built using Python and the Streamlit library. It allows users to:
- Load time series data.
- Select models for forecasting (ARIMA, SARIMA, Random Forest, Gradient Boosting, LSTM, GRU, Prophet).
- View graphs of training and test data.
- Analyze results using metrics (RMSE, MAPE, R²).
- Download PDF reports that include graphs and metrics.

## File Structure
- **app.py**: The main Streamlit application file for loading data, configuring models, and generating reports.
- **Dockerfile.streamlit**: Dockerfile for creating the Streamlit application image.
- **docker-compose.yml**: A file for configuring containers using Docker Compose.
- **requirements.txt**: A list of Python dependencies for the project.
- **.dockerignore**: Exclude files when building a Docker image.
- **Forecatsting**: Research notebook and pre-testing models
## Key features
1. **Data loading**:
- Load time series from Yahoo Finance (automatic loading of stock data is supported).
- Display data in a table.

2. **Model selection**:
- ARIMA
- SARIMA
- Random Forest
- Gradient Boosting
- LSTM
- GRU
- Prophet

3. **Visualization**:
- Plot training and test data.
- Forecast graphs with real data overlay.

4. **Model evaluation**:
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² (R-squared)

5. **Report generation**:
- Download PDF report with graphs, metrics and results.

## Installation

### Local environment

1. Download the project folder and go to it via terminal:
```bash
cd time-series-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

### Docker
1. Create a Docker image:
```bash
docker-compose build
```
2. Run the container:
```bash
docker-compose up
```
3. The application will be available at http://localhost.

## Usage
- Select data to load or load your own.
- Select a model and configure its parameters.
- View forecast results and metrics.
- Download a PDF report.

## Dependencies
The list of libraries is in `requirements.txt`, the main ones are:

- yfinance: Loading financial data.
- prophet: Time series forecasting.
- tensorflow: Implementation of LSTM and GRU models.
- pmdarima: Support for ARIMA and SARIMA.
- streamlit: Interactive web application.
- plotly: Interactive graphs.
- fpdf: Generation of PDF reports.

## Development
- `app.py`: main application file.
- `Dockerfile.streamlit`: Docker image configuration.
- `docker-compose.yml`: Launching the project via Docker Compose.