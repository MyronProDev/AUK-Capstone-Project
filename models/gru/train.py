from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU


def train_gru(X_train, y_train, seq_length):
    """
    Тренування GRU-моделі для прогнозування часових рядів.

    Parameters:
    - X_train (numpy array): Навчальний набір даних у форматі 3D (samples, time_steps, features).
    - y_train (numpy array): Цільові значення для навчання.
    - seq_length (int): Довжина часової послідовності (кількість попередніх кроків для прогнозування).

    Returns:
    - model (Sequential): Навчена GRU-модель.
    """
    # Ініціалізація моделі
    model = Sequential([
        GRU(50, activation='relu', input_shape=(seq_length, 1)),  # GRU-шар з 50 нейронами та функцією активації 'relu'
        Dense(1)  # Вихідний шар з одним нейроном для прогнозування одного значення
    ])

    # Компіляція моделі з оптимізатором 'adam' та функцією втрат 'mse'
    model.compile(optimizer='adam', loss='mse')  # 'adam' забезпечує швидке і стабільне навчання

    # Навчання моделі
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)  # Навчання на даних з вказаною кількістю епох і батчів

    return model  # Повертаємо навчану модель
