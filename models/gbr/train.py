from sklearn.ensemble import GradientBoostingRegressor


def train_gradient_boosting(X_train, y_train):
    """
    Тренування Gradient Boosting Regressor для прогнозування часових рядів.

    Parameters:
    - X_train (DataFrame or numpy array): Матриця ознак для навчання моделі.
    - y_train (Series or numpy array): Цільові значення для навчання.

    Returns:
    - model (GradientBoostingRegressor): Навчена модель Gradient Boosting.
    """
    # Ініціалізація моделі Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    # n_estimators=100: кількість дерев у ансамблі
    # random_state=42: фіксований генератор випадкових чисел для відтворюваності результату

    # Навчання моделі на основі навчальних даних
    model.fit(X_train, y_train)  # Підгонка моделі до навчальних даних

    return model  # Повертаємо навчану модель
