import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from metrics import MAE, MSE, RMSE, MAPE
from torch.utils.data import TensorDataset, DataLoader


def evaluate_model(model, X_test, y_test, scaler_y, configs):
    device = next(model.parameters()).device
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs.batch_size,
        shuffle=False)

    y_pred_list = []
    y_true_list = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(targets.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    y_true = y_true.reshape(-1, configs.pred_len, 5)
    y_pred = y_pred.reshape(-1, configs.pred_len, 5)
    print(f"y_true: {X_test.shape}")
    print(f"y_pred: {y_test.shape}")

    def inverse_scale(data):
        original_shape = data.shape
        return scaler_y.inverse_transform(
            data.reshape(-1, 5)
        ).reshape(original_shape)

    y_pred_inv = inverse_scale(y_pred)
    y_true_inv = inverse_scale(y_true)

    print("\n" + "="*50)
    print("Normalization")
    print(f"MAE: {MAE(y_pred, y_true):.8f}")
    print(f"MSE: {MSE(y_pred, y_true):.8f}")
    print(f"RMSE: {RMSE(y_pred, y_true):.8f}")
    print(f"MAPE: {MAPE(y_pred, y_true):.8f}%")

    print("\n" + "="*50)
    print("Inverse Normalization")
    print(f"MAE: {MAE(y_pred_inv, y_true_inv):.8f} V")
    print(f"MSE: {MSE(y_pred_inv, y_true_inv):.8f} V²")
    print(f"RMSE: {RMSE(y_pred_inv, y_true_inv):.8f} V")
    print(f"MAPE: {MAPE(y_pred_inv, y_true_inv):.8f}%")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'


    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(y_true_inv[10:58, 0, i].flatten(), label="True", color="blue", linewidth=2.5)
        plt.plot(y_pred_inv[10:58, 0, i].flatten(), label="Predicted", color="red", linestyle="--", linewidth=2.0)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{configs.best_model_path}.png", dpi=800)