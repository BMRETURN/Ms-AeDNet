import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(configs):
    input_volt_features = ['U1 (V)', 'U2 (V)', 'U3 (V)', 'U4 (V)', 'U5 (V)']
    input_factor_features = [
        'Time (h)', 'I (A)', 'TinH2 (C)', 'ToutH2 (C)', 'TinAIR (C)',
        'ToutAIR (C)', 'TinWAT (C)', 'ToutWAT (C)', 'PinAIR (mbara)',
        'PoutAIR (mbara)', 'PinH2 (mbara)', 'PoutH2 (mbara)', 'DinH2 (l/mn)',
        'DoutH2 (l/mn)', 'DinAIR (l/mn)', 'DoutAIR (l/mn)', 'DWAT (l/mn)',
        'HrAIRFC (%)'
    ]
    output_volt_features = ['U1 (V)', 'U2 (V)', 'U3 (V)', 'U4 (V)', 'U5 (V)']
    data = pd.read_csv(configs.data_path)

    # （6:2:2）
    total_samples = len(data)
    train_end = int(0.6 * total_samples)
    val_end = train_end + int(0.2 * total_samples)
    test_end = val_end + int(0.2 * total_samples)

    train_data = data.iloc[:train_end]

    scaler_x_volt = StandardScaler()
    scaler_x_factor = StandardScaler()
    scaler_y = StandardScaler()
    volt = data[input_volt_features].values

    X_train_volt = train_data[input_volt_features].values
    X_train_factor = train_data[input_factor_features].values
    y_train = train_data[output_volt_features].values

    scaler_x_volt.fit(volt)
    scaler_x_factor.fit(X_train_factor)
    scaler_y.fit(volt)

    X_all_volt = scaler_x_volt.transform(data[input_volt_features].values)
    X_all_factor = scaler_x_factor.transform(data[input_factor_features].values)
    X_all = np.hstack([X_all_volt, X_all_factor])

    y_all = scaler_y.transform(data[output_volt_features].values)

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:test_end], y_all[val_end:test_end]

    def create_windows(X, y):
        X_windows = []
        y_windows = []

        for i in range(len(X) - configs.seq_len - configs.pred_len + 1):
            X_windows.append(X[i:i + configs.seq_len])
            y_windows.append(y[i + configs.seq_len:i + configs.seq_len + configs.pred_len])
        return np.array(X_windows), np.array(y_windows)

    X_train, y_train = create_windows(X_train, y_train)
    X_val, y_val = create_windows(X_val, y_val)
    X_test, y_test = create_windows(X_test, y_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_x_volt, scaler_y