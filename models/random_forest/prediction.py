def predict_random_forest(model, X_test):
    """
    Прогнозування значень на основі навченої моделі Random Forest.

    Parameters:
    - model (RandomForestRegressor): Навчена модель Random Forest.
    - X_test (DataFrame or numpy array): Тестові дані для прогнозування.

    Returns:
    - numpy array: Масив прогнозованих значень.
    """
    # Виконуємо прогноз на основі тестових даних
    return model.predict(X_test)  # Повертаємо прогнозовані значення у форматі numpy array
