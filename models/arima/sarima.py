from pmdarima import auto_arima
from pmdarima.arima import ndiffs


def train_sarima_model(df_train):
    """
    Initialize an ARIMA model with automatic parameter selection.

    Parameters:
        df_train (pd.DataFrame): The training dataset containing the time series.

    Returns:
        model: Fitted seasonal ARIMA model.
        """
    # Визначення кількості диференціювань для ARIMA
    adf_diffs = ndiffs(df_train['rate'], alpha=0.05, test='adf',
                       max_d=6)  # Тест Дікі-Фуллера для визначення кількості диференціювань
    kpss_diffs = ndiffs(df_train['rate'], alpha=0.05, test='kpss',
                        max_d=6)  # Тест KPSS для визначення кількості диференціювань
    n_diffs = max(adf_diffs, kpss_diffs)  # Вибираємо максимальну кількість диференціювань

    model_sarima = auto_arima(df_train.rate,  # Цільова змінна для навчання
                              d=n_diffs,  # Кількість диференціювань для досягнення стаціонарності
                              seasonal=True,  # Увімкнено сезонну компоненту
                              m=3,  # Сезонний період (наприклад, 3 для квартальних даних)
                              stepwise=True,  # Покроковий підбір параметрів для прискорення обчислень
                              suppress_warnings=True,  # Вимикає попередження під час навчання моделі
                              error_action="ignore",  # Ігнорує помилки в процесі підбору параметрів
                              max_p=6,  # Максимальне значення для параметра p (авторегресивний порядок)
                              max_order=None,  # Не обмежує сумарний порядок моделі (p+q)
                              trace=False  # Вимикає вивід процесу підбору параметрів
                              )

    return model_sarima
