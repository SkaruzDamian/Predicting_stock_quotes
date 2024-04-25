import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time

def downloadingFilesFromWeb():
    download_dir = 'D:\\Notowania'
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing_for_trusted_sources_enabled": False,
        "safebrowsing.enabled": False
    })
    driver = webdriver.Chrome(options=chrome_options)
    url = 'https://www.gpw.pl/archiwum-notowan?fetch=0&type=10&instrument=&date=&show_x=Pokaż+wyniki'
    driver.get(url)
    time.sleep(10)
    select_element = driver.find_element(By.XPATH, "//select[@name='instrument']")
    select = Select(select_element)
    options = select.options
    for option in options[1:]:
        select.select_by_value(option.get_attribute('value'))
        time.sleep(10)
        download_button = driver.find_element(By.XPATH, "//a[contains(@onclick, 'downloadXLS')]")
        driver.execute_script("arguments[0].scrollIntoView(true);", download_button)
        driver.execute_script("arguments[0].click();", download_button)
        time.sleep(10)
    driver.quit()

def process_downloaded_files(download_dir):
    file_data = []
    output_dir = 'D:\\Notowania'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if filename.endswith('.xls'):
            data = pd.read_excel(file_path)
            if 'Data' in data.columns:
                data['Data'] = pd.to_datetime(data['Data'])
                max_year = data['Data'].dt.year.max()
                if max_year != 2024:
                    continue
                data_weekly = data.resample('W-Mon', on='Data').agg({
                    'Kurs otwarcia': 'first',
                    'Kurs zamknięcia': 'last',
                    'Wolumen': 'sum'
                }).reset_index()
                new_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_weekly.xls")
                data_weekly.to_excel(new_file_path, index=False)
                last_close_price = data['Kurs zamknięcia'].iloc[-1]
                file_data.append((file_path, last_close_price))
                os.remove(file_path)
    sorted_files = sorted(file_data, key=lambda x: x[1], reverse=True)
    sorted_files_paths = [file[0] for file in sorted_files]
    return sorted_files_paths

def Calculate():
    folder_path = 'D:\\NotowaniaTygodniowe'
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xls"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path)
            data = df.iloc[-2300:]
            new_dataset = pd.DataFrame(index=range(0, len(data)), columns=['Data', 'Kurs zamknięcia'])
            new_dataset["Data"] = data['Data'].values
            new_dataset["Kurs zamknięcia"] = data["Kurs zamknięcia"].values
            new_dataset.set_index("Data", inplace=True)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(new_dataset)
            x_train_data, y_train_data = [], []
            for i in range(1, len(scaled_data) - 5):
                x_train_data.append(scaled_data[i - 1:i, 0])
                y_train_data.append(scaled_data[i:i + 5, 0])
            x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
            x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=350, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
            lstm_model.add(LSTM(units=350))
            lstm_model.add(Dense(5))
            lstm_model.compile(loss='mean_squared_error', optimizer='adam')
            lstm_model.fit(x_train_data, y_train_data, epochs=300, batch_size=1, verbose=0)
            predicted_closing_price = lstm_model.predict(x_train_data[-1].reshape(1, x_train_data.shape[1], 1))
            predicted_closing_price = scaler.inverse_transform(predicted_closing_price)
            print(f"Przewidywane wartości dla akcji {file_name}:")
            print(predicted_closing_price.flatten())

def MainMethod():
    folder_path = 'D:\\Notowania'
    while True:
        print("Podaj liczbę i stwierdź co chcesz zrobić?")
        print("1. Pobierz dane z sieci")
        print("2. Przerób na interwał tygodniowy i posortuj malejąco")
        print("3. Oblicz klasyfikacje oraz metode szczytów i dołków")
        print("4. Wyjdź z programu")
        choice = int(input("Podaj liczbę"))
        match choice:
            case 1:
                downloadingFilesFromWeb()
            case 2:
                process_downloaded_files(folder_path)
            case 3:
                Calculate()
            case 4:
                break
            case _:
                print("Zły wybór")

MainMethod()
