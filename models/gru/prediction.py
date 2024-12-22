

def predict_gru(model, X_test, scaler):
    """
    Прогнозування на основі навченої GRU-моделі з денормалізацією результатів.

    Parameters:
    - model (Sequential): Навчена GRU-модель.
    - X_test (numpy array): Тестові дані у форматі 3D (samples, time_steps, features).
    - scaler (MinMaxScaler): Масштабувальник, який використовувався для нормалізації навчальних даних.

    Returns:
    - predictions (numpy array): Денормалізовані прогнозовані значення у форматі 1D.
    """
    # Виконання прогнозу на тестових даних
    predictions = model.predict(X_test)  # Отримуємо прогнозовані значення

    # Денормалізація прогнозованих значень за допомогою масштабувальника
    return scaler.inverse_transform(predictions).flatten()  # Перетворюємо назад у початковий масштаб та "сплющуємо" у 1D