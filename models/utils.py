import numpy as np
from pytickersymbols import PyTickerSymbols


def create_lag_features(data, lags=5):
    """
    Створює лагові ознаки для часових рядів.

    Parameters:
    - data (DataFrame): Вхідний датафрейм з колонкою 'rate'.
    - lags (int): Кількість лагів, які потрібно створити.

    Returns:
    - DataFrame: Датафрейм з доданими лаговими ознаками.
    """
    for lag in range(1, lags + 1):  # Цикл по кількості лагів
        data[f'lag_{lag}'] = data['rate'].shift(lag)  # Створюємо лагову ознаку для кожного лагу
    return data  # Повертаємо датафрейм з лаговими о


def create_sequences(data, seq_length):
    """
    Створює послідовності для навчання моделей LSTM/GRU.

    Parameters:
    - data (numpy array): Масштабовані дані у форматі numpy array.
    - seq_length (int): Довжина послідовності.

    Returns:
    - X, y: Масиви для навчання у форматі (samples, time_steps, features).
    """
    X, y = [], []  # Ініціалізуємо списки для послідовностей
    for i in range(len(data) - seq_length):  # Цикл по всім можливим послідовностям
        X.append(data[i:i + seq_length, 0])  # Додаємо вхідну послідовність
        y.append(data[i + seq_length, 0])  # Додаємо цільове значення
    return np.array(X), np.array(y)  # Повертаємо X і y у форматі numpy array


def get_tickers():
    stock_data = PyTickerSymbols()
    nasdaq_tickers = stock_data.get_stocks_by_index('S&P 500')  # Corrected index name
    _list = []
    # Print the ticker symbols
    for stock in nasdaq_tickers:
        _list.append(stock['symbol'])

    return _list
