import numpy as np


# Функція для прогнозування одного кроку вперед
def forecast_one_step(model):
    """
    Виконує прогноз на 1 крок вперед і повертає значення прогнозу та довірчий інтервал.

    Returns:
    - fc (float): Прогнозоване значення.
    - conf_int (list): Довірчий інтервал для прогнозу.
    """
    fc, conf_int = model.predict(n_periods=1,  # Прогноз на 1 період
                                 return_conf_int=True)  # Повертає довірчий інтервал
    return (
        fc.tolist()[0],  # Повертаємо перше значення прогнозу як float
        np.asarray(conf_int).tolist()[0]  # Конвертуємо довірчий інтервал у список
    )


def predict_date_by_date(model, df_test, ticker):
    """
    Perform day-by-day prediction using an ARIMA model, updating the model with each observation.

    Parameters:
        model: The fitted ARIMA model.
        df_test (pd.DataFrame): The test dataset containing the time series.

    Returns:
        tuple: A tuple containing:
            - y_predict_day_by_day (list): Predicted values for each step.
            - y_confidence (list): Confidence intervals for each prediction.
    """
    # Списки для збереження прогнозованих значень та довірчих інтервалів
    y_predict_day_by_day = []  # Зберігає прогнозовані значення на кожному кроці
    y_confidence = []  # Зберігає довірчі інтервали для прогнозу

    # Цикл для послідовного прогнозування та оновлення моделі
    for i, new_ob in enumerate(df_test[ticker].values):  # Ітеруємо по всім значенням тестового набору
        fc, conf = forecast_one_step(model=model)  # Виконуємо прогноз на один крок вперед
        y_predict_day_by_day.append(fc)  # Додаємо прогнозоване значення до списку
        y_confidence.append(conf)  # Додаємо довірчий інтервал до списку

        model.update(new_ob)  # Оновлюємо ARIMA модель новим спостереженням з тестових даних

    return y_predict_day_by_day
