import pandas as pd


def predict_prophet(model, df_test, y_test_ml):
    """
    Виконує прогнозування за допомогою навченої моделі Prophet.

    Parameters:
    - model (Prophet): Навчена модель Prophet.
    - df_test (DataFrame): Тестовий набір даних з індексом, що містить дати для прогнозу.

    Returns:
    - prophet_forecast (numpy array): Прогнозовані значення (yhat) для тестового набору даних.
    """
    # Створення датафрейму для майбутніх прогнозів на основі дат із тестового набору
    future = pd.DataFrame(df_test.index, columns=['ds'])  # Створюємо датафрейм із датами для прогнозування

    # Генеруємо майбутній датафрейм на вказану кількість періодів
    future = model.make_future_dataframe(periods=len(df_test))  # Додаємо періоди для тестових даних

    # Виконання прогнозування
    forecast_prophet = model.predict(future)  # Отримуємо прогнозовані значення

    # Вибираємо останні прогнозовані значення (yhat), що відповідають тестовому набору
    prophet_forecast = forecast_prophet['yhat'].tail(len(y_test_ml)).values

    return prophet_forecast  # Повертаємо прогнозовані значення