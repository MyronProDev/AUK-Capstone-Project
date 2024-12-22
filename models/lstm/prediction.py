

def predict_lstm(model, X_test, scaler):
    """
    Виконує прогнозування за допомогою навченої LSTM-моделі та денормалізує результати.

    Parameters:
    - model (Sequential): Навчена LSTM-модель.
    - X_test (numpy array): Тестовий набір даних у форматі 3D (samples, time_steps, features).
    - scaler (MinMaxScaler): Масштабувальник, що використовувався для нормалізації даних.

    Returns:
    - numpy array: Денормалізовані прогнозовані значення у форматі 1D.
    """
    # Виконання прогнозування на тестових даних
    predictions = model.predict(X_test)  # Генеруємо прогнозовані значення

    # Денормалізація прогнозів, щоб повернути їх у початковий масштаб
    return scaler.inverse_transform(predictions).flatten()  # Перетворюємо результат у 1D масив