import sys
sys.path.insert(0, '/share/jobdata/d0122001101428/H2/PEMFC_RUL')
from data_loader import load_data
from train import train_model
from evaluate import evaluate_model
import torch
import random
import numpy as np
import argparse
from models.MsAeDNet import MsAeDNet
from models.DLinear import DLinear
from models.Transformer import Transformer
from models.GRU import GRU
from models.LSTM import LSTM
from models.TCN import TCN
from models.LightTS import LightTS
from models.PatchTST import PatchTST


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="config")

    parser.add_argument('--data_path', type=str, default="input/FC2_With_Ripples_Excel.csv")
    # FC1_Without_Ripples_Excel  FC2_With_Ripples_Excel
    parser.add_argument('--data', type=str, default="FC2")
    parser.add_argument('--model_name', type=str, default="MsAeDNet")
    # MsAeDNet  DLinear  Transformer  GRU  LSTM  TCN  LightTS  PatchTST
    parser.add_argument('--model_save_path', type=str, default="prediction/")

    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--patch_size', type=int, default=6)
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--point_dim', type=int, default=64)
    parser.add_argument('--patch_dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=5)

    # DLinear / former
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layer', type=int, default=2)


    configs = parser.parse_args()
    best_model_path = f"{configs.model_save_path}{configs.model_name}_{configs.seq_len}_{configs.pred_len}_{configs.data}"
    configs.best_model_path = best_model_path
    print(configs.model_name)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler_x_volt, scaler_y = load_data(configs)
    if configs.model_name == 'MsAeDNet':
        model = MsAeDNet(configs)
    elif configs.model_name == 'DLinear':
        model = DLinear(configs)
    elif configs.model_name == 'Transformer':
        model = Transformer(configs)
    elif configs.model_name == 'GRU':
        model = GRU(configs)
    elif configs.model_name == 'LSTM':
        model = LSTM(configs)
    elif configs.model_name == 'TCN':
        model = TCN(configs)
    elif configs.model_name == 'LightTS':
        model = LightTS(configs)
    elif configs.model_name == 'PatchTST':
        model = PatchTST(configs)
    else:
        print('model name error!')

    model = train_model(model, X_train, y_train, X_val, y_val, configs)

    model.load_state_dict(torch.load(f"{configs.best_model_path}.pth"))

    evaluate_model(model, X_test, y_test, scaler_y, configs)