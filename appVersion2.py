import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tensorflow.python.client import device_lib
from numba import jit, cuda
import tensorflow as tf




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
    similarity = similarity*100.0
    return similarity

tf.config.run_functions_eagerly(True)
@tf.function
def calculate():
    lstm_units_list = [1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,26,27,28,29, 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49 ,50,50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    sequence_length_list = [5, 10, 15, 20,25,30,60,90]
    epochs_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39, 40,41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    values_list = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600]
    batch_sizes = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128 ]
    learning_rates = [0.001, 0.01]
    optimizers = {'adam': Adam, 'sgd': SGD, 'rmsprop': RMSprop}
    regularizations = [(0.0001, 0.0001),(0.001,0.001),(0.01, 0.01), (0.1, 0.1)]  
    patiences = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Denses = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512]

    file_path = 'D:\\Notowania\\eurocash_to_2024-04-26_akcje.xls'
    df = pd.read_excel(file_path)
    scaler = StandardScaler()
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
                        for pat in patiences:
                            for den in Denses:
                                for optimizer_key, optimizer in optimizers.items():
                                    for lr in learning_rates:
                                        for batch_size in batch_sizes:
                                            for reg in regularizations:
                                                data = df['Kurs zamknięcia'].values[-value_count:]
                                                data_scaled = scaler.transform(data.reshape(-1, 1))

                                                X, y = [], []
                                                for i in range(len(data_scaled) - sequence_length):
                                                    X.append(data_scaled[i:i + sequence_length])
                                                    y.append(data_scaled[i + sequence_length])
                                                X = np.array(X).reshape((len(X), sequence_length, 1))
                                                y = np.array(y).reshape((len(y), 1))

                                                
                                                model = Sequential([
                                                        LSTM(lstm_units, activation='tanh', input_shape=(sequence_length, 1),
                                                            kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]), return_sequences=True),
                                                        Dropout(drop),
                                                        Dense(den, activation='tanh'),
                                                        Dense(1)
                                                   
                                                    ])

                                                opt = optimizer(learning_rate=lr)
                                                model.compile(optimizer=opt, loss='mse')

                                                early_stopping = EarlyStopping(monitor='val_loss', patience=pat, restore_best_weights=True)

                                                model.fit(X, y, epochs=epoch, batch_size=batch_size,
                                                        validation_data=(X, y), callbacks=[early_stopping], verbose=0)

                                                last_five_predictions_inversed = []
                                                last_five_predictions = []
                                                
                                                next_value_scaled = model.predict(X[-1].reshape(1, sequence_length, 1))
                                                data_scaled = np.append(data_scaled, next_value_scaled)
                                                X[-1] = np.append(X[-1][1:], next_value_scaled[0][0]).reshape(sequence_length, 1)
                                                last_five_predictions.append(next_value_scaled)


                                                last_five_predictions_inversed = scaler.inverse_transform(np.array(last_five_predictions).reshape(-1, 1)).flatten()
                                                similarity = compare_arrays(last_five_predictions_inversed, actual_values)
                                                print(f"Najlepsze parametry: {best_params}")
                                                print(f"Najlepsza dokladnosc: {max_similarity}")
                                                print(f"Najlepsze przewidywania: {best_predictions}")
                                                print(f"Ostatnie 5 prognoz: {last_five_predictions_inversed}")
                                                print(f" Obecne parametry{lstm_units, optimizer_key, lr, batch_size, reg, sequence_length, epoch, value_count, drop}")
                                                if similarity >= 98.0:
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
                                        
                                            
calculate()
