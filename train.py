import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

def train_model(model, X_train, y_train, X_val, y_val, configs):
    device = torch.device(f'cuda:{configs.gpu}' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr)
    criterion = nn.L1Loss()

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True)

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()  # 记录开始时间

    for epoch in range(configs.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - epoch_start
                print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Avg Loss: {total_loss / (batch_idx + 1):.4f} | "
                      f"Elapsed: {elapsed:.1f}s")

        epoch_time = time.time() - epoch_start

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                total_val_loss += criterion(outputs, targets).item()
            val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{configs.epochs} | "
              f"Time: {epoch_time // 60:.0f}m{epoch_time % 60:.0f}s | "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{configs.best_model_path}.pth")
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 计算时间指标
    total_time = time.time() - start_time
    total_epochs = epoch + 1  # 实际运行的epoch数（从0开始计数）
    total_batches = total_epochs * len(train_loader)
    avg_time_per_batch = total_time / total_batches if total_batches != 0 else 0

    print(f"\n[效率报告] 总训练时间: {total_time:.2f}s | 平均每batch耗时: {avg_time_per_batch:.4f}s")
    return model