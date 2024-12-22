def predict_gradient_boosting(model, X_test):
    """
    Прогнозування значень на основі навченої моделі Gradient Boosting.

    Parameters:
    - model (GradientBoostingRegressor): Навчена модель Gradient Boosting.
    - X_test (DataFrame or numpy array): Тестові дані (матриця ознак) для прогнозування.

    Returns:
    - numpy array: Масив прогнозованих значень.
    """
    return model.predict(X_test)  # Виконуємо прогноз на тестових даних