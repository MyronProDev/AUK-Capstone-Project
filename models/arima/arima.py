from pmdarima import auto_arima
from pmdarima.arima import ndiffs


def train_arima_model(df_train):
    """
    Initialize an ARIMA model with automatic parameter selection.

    Parameters:
        df_train (pd.DataFrame): The training dataset containing the time series.

    Returns:
        model: Fitted ARIMA model.
    """
    # Визначення кількості диференціювань для ARIMA
    adf_diffs = ndiffs(df_train['rate'], alpha=0.05, test='adf',
                       max_d=6)  # Тест Дікі-Фуллера для визначення кількості диференціювань
    kpss_diffs = ndiffs(df_train['rate'], alpha=0.05, test='kpss',
                        max_d=6)  # Тест KPSS для визначення кількості диференціювань
    n_diffs = max(adf_diffs, kpss_diffs)  # Вибираємо максимальну кількість диференціювань
    # print(f"Optimal number of differences: {n_diffs}")  # Виводимо результат

    # Ініціалізація ARIMA моделі з автоматичним підбором параметрів
    model = auto_arima(df_train.rate,  # Цільова змінна: ряд для навчання
                       d=n_diffs,  # Параметр диференціювання для стаціонарності
                       seasonal=False,  # Вимикаємо сезонність (якщо потрібно сезонність, встановіть True)
                       suppress_warnings=True,  # Вимикає попередження під час роботи
                       error_action="ignore",  # Ігнорує можливі помилки у процесі підбору
                       max_p=6,  # Максимальне значення для параметра p (авторегресивний порядок)
                       max_order=None,  # Не обмежує сумарний порядок моделі (p+q)
                       trace=False  # Вимикає виведення проміжних результатів підбору параметрів
                       )
    return model
