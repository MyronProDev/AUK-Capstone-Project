from prophet import Prophet


def train_prophet(df_train):
    """
    Тренування моделі Prophet для прогнозування часових рядів.

    Parameters:
    - df_train (DataFrame): Навчальний набір даних з колонками 'date' і 'rate'.

    Returns:
    - model (Prophet): Навчена модель Prophet.
    """
    # Підготовка даних для Prophet: перетворюємо формат даних
    df_train_prophet = df_train.reset_index()[['date', 'rate']]  # Скидаємо індекс та вибираємо потрібні колонки
    df_train_prophet.columns = ['ds', 'y']  # Prophet вимагає назви колонок: 'ds' (дата) та 'y' (значення)

    # Ініціалізація та навчання моделі Prophet
    model = Prophet()  # Ініціалізуємо об'єкт моделі Prophet
    model.fit(df_train_prophet)  # Навчаємо модель на підготовлених даних

    return model  # Повертаємо навчану модель
