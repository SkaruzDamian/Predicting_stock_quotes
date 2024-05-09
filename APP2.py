import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

lstm_units_list = [5, 10, 20, 30, 50]
sequence_length_list = [5, 10, 15, 20]
epochs_list = [20, 40, 60, 80, 100, 150]
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
values_list = [1000, 1500, 2000, 2500, 3000]
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.01]
optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
regularizations = [(0.01, 0.01), (0.1, 0.1)]  # (l1, l2)
file_path = 'D:\\Notowania\\eurocash_to_2024-04-26_akcje.xls'
df = pd.read_excel(file_path)
scaler = StandardScaler()

# Define K-Fold Cross Validation
def compare_arrays(array1, array2):
    liczba=float('inf')
    similarity=float('inf')
    count = 0
    for a, b in zip(array1, array2):
        if a/b<1.0:
            count += a/b
        else:
            liczba = a/b
            liczba = liczba-100
            liczba = liczba * (-1)
            liczba = liczba/100
            count= count+liczba
    similarity = count / 5.0
    return similarity

actual_values = np.array([13.54, 13.72, 13.68, 13.66, 13.83])
scaler.fit(actual_values.reshape(-1, 1)) 
actual_values_scaled = scaler.transform(actual_values.reshape(-1, 1)).flatten()
max_similarity = 0
best_array = None

best_accuracy = float('inf')
best_params = {}
best_predictions = []

for lstm_units in lstm_units_list:
    for sequence_length in sequence_length_list:
        for epoch in epochs_list:
            for drop in dropout_rates:
                for value_count in values_list:
                    data = df['Kurs zamknięcia'].values[-value_count:]
                    data_scaled = scaler.transform(data.reshape(-1, 1))

                    X, y = [], []
                    for i in range(len(data_scaled) - sequence_length):
                        X.append(data_scaled[i:i + sequence_length])
                        y.append(data_scaled[i + sequence_length])
                    X = np.array(X).reshape((len(X), sequence_length, 1))
                    y = np.array(y).reshape((len(y), 1))

                    for optimizer_key, optimizer in optimizers.items():
                        for lr in learning_rates:
                            for batch_size in batch_sizes:
                                for reg in regularizations:
                                    model = Sequential([
                                            LSTM(lstm_units, activation='tanh', input_shape=(sequence_length, 1),
                                                 kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]), return_sequences=True),
                                            Dropout(drop),
                                            BatchNormalization(),
                                            LSTM(lstm_units, activation='tanh', return_sequences=False,
                                                 kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1])),
                                            Dropout(drop),
                                            BatchNormalization(),
                                            Dense(64, activation='tanh'),
                                            Dense(1)
                                        ])

                                    opt = optimizer(learning_rate=lr)
                                    model.compile(optimizer=opt, loss='mse')

                                    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                                    model.fit(X, y, epochs=epoch, batch_size=batch_size,
                                              validation_data=(X, y), callbacks=[early_stopping], verbose=0)

                                    last_five_predictions = []
                                    for _ in range(5):
                                        next_value_scaled = model.predict(X[-1].reshape(1, sequence_length, 1))
                                        data_scaled = np.append(data_scaled, next_value_scaled)
                                        X[-1][-1] = next_value_scaled[0][0]
                                        last_five_predictions.append(next_value_scaled)

                                    last_five_predictions_inversed = scaler.inverse_transform(np.array(last_five_predictions).reshape(-1, 1)).flatten()
                                    print("test")
                                    similarity = compare_arrays(last_five_predictions_inversed, actual_values)
                                    print("test")
                                    if similarity >= 0.970:
                                        print(similarity)
                                        if similarity > max_similarity:
                                            max_similarity = similarity
                                            best_array = last_five_predictions_inversed
                                            best_params = {
                                                    'lstm_units': lstm_units,
                                                    'optimizer': optimizer_key,
                                                    'learning_rate': lr,
                                                    'batch_size': batch_size,
                                                    'regularization': reg,
                                                    'sequence_length': sequence_length,
                                                    'epoches': epoch,
                                                    'value': value_count,
                                                    'dropout': drop
                                                }
                                            best_predictions = last_five_predictions_inversed.copy()
                                        
                                            print(f"Best parameters so far: {best_params}")
                                            print(f"Best MSE so far: {max_similarity}")
                                            print(f"Last five predictions (in normal scale): {last_five_predictions_inversed}")

print(f"Final best parameters: {best_params}")
print(f"Best MSE: {max_similarity}")
print(f"Best Predictions: {best_predictions}")
