import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
import argparse
from models import LSTM_layer
from torch.utils.data import TensorDataset, DataLoader
from module import MSE_loss, MSE_diff_loss, Trend_loss
from module import rmse, mae, mape
from module import Load_dataset


######################### Sliding Window Dataset #########################
def make_sliding_dataset(X_hist: np.ndarray,
                         y_hist: np.ndarray,
                         window: int = 1):

    T = len(X_hist)
    if T < window:
        return None, None

    X_list, y_list = [], []
    # Sliding based on index t where the window ends
    for t in range(window - 1, T):
        s = t - window + 1
        e = t + 1
        X_list.append(X_hist[s:e])
        y_list.append(y_hist[s:e])

    X_arr = np.stack(X_list, axis=0)
    y_arr = np.stack(y_list, axis=0)
    return X_arr, y_arr


######################### Train / Finetune / Eval #########################
def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for xb, yb in train_loader:
        xb = xb.cuda()
        yb = yb.cuda()
        optimizer.zero_grad()

        pred = model(xb)  
        mse_loss = MSE_loss(pred, yb)

        Loss_F = mse_loss

        Loss_F.backward()
        optimizer.step()

    if args.show_train_loss:

        print(
            f"[{epoch}/{args.max_epoch}] Loss-sum: {Loss_F.item():.4f}  "
            f"MSE: {mse_loss.item():.4f}"
        )


def train_ft(args, model, train_loader, optimizer, epoch):
    model.train()
    for xb, yb in train_loader:
        xb = xb.cuda()
        yb = yb.cuda()
        optimizer.zero_grad()

        pred = model(xb)
        mse_loss = MSE_loss(pred, yb)


        Loss_F = mse_loss

        Loss_F.backward()
        optimizer.step()

    if args.show_train_loss:

        print(
            f"[FT {epoch}/{args.ft_epoch}] Loss-sum: {Loss_F.item():.4f}  "
            f"MSE: {mse_loss.item():.4f}"
        )


def eval_once(args, X_test, model):
    model.eval()
    with torch.no_grad():
        pred = model(X_test.cuda())  
        pred_np = pred.cpu().numpy().reshape(-1, 1)
    return pred_np


############################### MAIN ################################
def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    ################ Load & Preprocess Data ################
    data_raw = Load_dataset(
        foundation_path="/home/bxai4/financial/FingFinge/Data"
    )
    # Forward fill & Inner join
    data = data_raw.ffill().dropna().reset_index(drop=True)
    data["target_shift"] = data["target"].shift(-1)

    # Remove target shift Nan(Last Raw)
    data = data.dropna(subset=["target_shift"]).reset_index(drop=True)

    raw = data.copy()
    # Control starting date 
    starting_point = raw[raw["orig_date"] == "2022-12-29"].index[0]

    feature_cols = [
        "VIX_Close",
        "KOSPI200_LogRet",
        "USD_KRW_LogRet",
        "DGS10",
        "target",
    ] #
    target_col = "target_shift"

    use_scaler = args.use_scaler  # X scaling true/false
    window = args.window_len      # Sliding window length 

    model = None
    optimizer = None
    scheduler = None
    first_train_done = False

    # Store predict/target
    all_preds = []
    all_trues = []
    forecast_list = []
    true_list = []
    ################ Walk-forward Loop ################
    for i in range(starting_point, len(raw) - 1):
        # 1) Past Interval (Include current i)
        past_df = raw.loc[:i].copy()

        X_hist = past_df[feature_cols].values           # (T, C)
        y_hist = past_df[target_col].values.reshape(-1, 1)  # (T, 1)

        # Creating a Sliding Window Dataset
        X_win_np, y_win_np = make_sliding_dataset(X_hist, y_hist, window=window)
        if X_win_np is None:
            # Skip interval that not satisfy window length
            continue

        # 2) Scaling(just x)
        if use_scaler:
            scaler_x = MinMaxScaler(feature_range=(-1, 1))
            # Fit about all window data
            T_all = X_win_np.reshape(-1, X_win_np.shape[-1])
            scaler_x.fit(T_all)
            X_win_np = scaler_x.transform(T_all).reshape(X_win_np.shape)
        else:
            scaler_x = None

        # 3) Transform tensor
        X_train = torch.tensor(X_win_np, dtype=torch.float32)   # (N, window, C)
        y_train = torch.tensor(y_win_np, dtype=torch.float32)   # (N, window, 1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        # 4) Model initialize
        if model is None:
            if args.model_name == "LSTM":
                model = LSTM_layer(
                    input_dim=X_train.shape[2],
                    d_model=args.d_model,
                    output_dim=y_train.shape[2],
                )
            model = nn.DataParallel(model, device_ids=args.gpuNum).cuda()

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.LR,
                betas=(0.8, 0.99),
                weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.999, last_epoch=-1
            )

        # 5) train / finetune
        if not first_train_done:
            print(f"-> First training at i={i}: epoch={args.max_epoch}, LR={args.LR}")
            for epoch in range(args.max_epoch):
                train(args, model, train_loader, optimizer, epoch)
                scheduler.step()
            first_train_done = True
        else:
            optimizer.param_groups[0]["lr"] = args.ft_LR
            print(f"-> Fine-tuning at i={i}: epoch={args.ft_epoch}, LR={args.ft_LR}")
            for epoch in range(args.ft_epoch):
                train_ft(args, model, train_loader, optimizer, epoch)

        # 6) For predict X_test
        # --- one-step ahead 예측 ---
        test_df = raw.loc[:i+1].copy()  # i까지의 feature만 사용
        X_test_seq = test_df[feature_cols].values
        if use_scaler:
            X_test_seq = scaler_x.transform(X_test_seq)
    
        X_test = torch.tensor(X_test_seq, dtype=torch.float32).unsqueeze(0)
        forecast_array = eval_once(args, X_test, model)  # (T_test, 1)
    
        y_hat = float(forecast_array[-1, 0])          # scalar
        y_true = float(raw.loc[i+1, target_col])        # target_shift[i] = target[i+1]
    
        forecast_list.append(y_hat)
        true_list.append(y_true)

        forecast_rmse = rmse(np.array(true_list), np.array(forecast_list))

        # Append Date
        if i + 2 > len(raw) - 1:
            forecast_date = str(raw.iloc[-1]["orig_date"])
        else:
            forecast_date = str(raw.loc[i + 2, "orig_date"])

        print(
            f"Model-Name: {args.model_name}\t "
            f"forecast_date: {forecast_date}\t "
            f"RMSE: {forecast_rmse:.4f}"
        )
    all_preds = pd.DataFrame(np.array(forecast_list))
    # 8) Store last predicted interval 
    joblib.dump(all_preds, f"{args.model_name}_result_no_target.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--model_name", type=str, default="LSTM", help="LSTM GRU TimesGPT")
    parser.add_argument("--gpuNum", type=list, default=[0])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--LR", type=float, default=1e-4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--show_train_loss", type=bool, default=False)
    parser.add_argument("--use_scaler", type=bool, default=True)
    parser.add_argument("--ft_epoch", type=int, default=50)
    parser.add_argument("--ft_LR", type=float, default=1e-5)
    parser.add_argument("--window_len", type=int, default=300)
    parser.add_argument("--d_model", type=int, default=64)

    args = parser.parse_args()
    main(args)
