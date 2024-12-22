from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU


def train_lstm(X_train, y_train, seq_length):
    """
    Тренування LSTM-моделі для прогнозування часових рядів.

    Parameters:
    - X_train (numpy array): Навчальний набір даних у форматі 3D (samples, time_steps, features).
    - y_train (numpy array): Цільові значення для навчання.
    - seq_length (int): Довжина часової послідовності (кількість попередніх кроків часу для прогнозу).

    Returns:
    - model (Sequential): Навчена LSTM-модель.
    """
    # Ініціалізація нейронної мережі
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),  # Додаємо LSTM-шар з 50 нейронами і ReLU активацією
        Dense(1)  # Вихідний шар з одним нейроном для прогнозування одного значення
    ])

    # Компіляція моделі з оптимізатором 'adam' та функцією втрат 'mse'
    model.compile(optimizer='adam', loss='mse')  # Adam – ефективний оптимізатор, MSE – функція втрат для регресії

    # Навчання моделі на навчальних даних
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    # epochs=20: кількість ітерацій по всьому набору даних
    # batch_size=16: розмір підвибірки даних на одну ітерацію

    return model  # Повертаємо навчану модель
