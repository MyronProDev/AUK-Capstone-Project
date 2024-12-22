from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train):
    """
    Тренування моделі Random Forest для прогнозування часових рядів.

    Parameters:
    - X_train (DataFrame or numpy array): Матриця ознак для навчання.
    - y_train (Series or numpy array): Цільові значення для навчання.

    Returns:
    - model (RandomForestRegressor): Навчена модель Random Forest.
    """
    # Ініціалізація Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # n_estimators=100: кількість дерев у лісі (більше значення — краща стабільність, але більше обчислень)
    # random_state=42: фіксує випадковий генератор для відтворюваності результатів

    # Навчання моделі на навчальних даних
    model.fit(X_train, y_train)  # Підгонка моделі до навчальних даних (X_train: ознаки, y_train: цільова змінна)

    return model  # Повертаємо навчану модель