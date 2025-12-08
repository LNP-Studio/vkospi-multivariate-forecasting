import os
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import torch
import torch.nn.functional as F

def Load_dataset(foundation_path, start="2004-01-02", end="2025-09-30"):
    ################################## Load target ##################################
    target = pd.read_csv(os.path.join(foundation_path, "VKOSPI_2004_2025_CL.csv"))
    
    # Load fx (USD/KRW 환율 레벨)
    fx = web.DataReader('DEXKOUS', 'fred', start=start, end=end)
    fx = fx.rename(columns={'DEXKOUS': 'USD_KRW'})
    fx.reset_index(inplace=True)  # 'DATE' 컬럼 생김
    
    ################################## Load Kospi ##################################
    kospi200_ticker = '^KS200'
    kospi_raw = yf.download(kospi200_ticker, start=start, end=end)
    
    if isinstance(kospi_raw.columns, pd.MultiIndex):
        k_close = kospi_raw.xs('Close', axis=1, level=0)
        if isinstance(k_close, pd.DataFrame):
            k_close = k_close.iloc[:, 0]
        kospi = pd.DataFrame({'KOSPI200': k_close})
    else:
        kospi = kospi_raw[['Close']].rename(columns={'Close': 'KOSPI200'})
    
    kospi.reset_index(inplace=True)  # Date 컬럼 생김
    
    ################################## Load VIX ##################################
    vix_raw = yf.download('^VIX', start=start, end=end)
    
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw.xs('Close', axis=1, level=0)
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
        vix = pd.DataFrame({'VIX_Close': vix_close})
    else:
        vix = vix_raw[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    vix.reset_index(inplace=True)
    
    ################################# Load Rate ###################################
    DGS = pd.read_csv(os.path.join(foundation_path, "DGS10_2004_2025.csv"))
    
    ################################## Ensure Date is datetime ##################################
    DGS['Date']   = pd.to_datetime(DGS['Date'])
    target['Date'] = pd.to_datetime(target['Date'])
    vix['Date']   = pd.to_datetime(vix['Date'])
    kospi['Date'] = pd.to_datetime(kospi['Date'])
    fx['Date']    = pd.to_datetime(fx['DATE'])   # FRED 'DATE' → 'Date'
    
    ################################## Set Date as index ##################################
    DGS   = DGS.set_index("Date")
    target = target.set_index("Date")
    vix   = vix.set_index("Date")
    kospi = kospi.set_index("Date")
    fx    = fx.set_index("Date")
    
    ################################## target 인덱스 기준 align ##################################
    DGS   = DGS.reindex(target.index)
    vix   = vix.reindex(target.index)
    kospi = kospi.reindex(target.index)
    fx    = fx.reindex(target.index)
    
    ################################## concat 이후 ##################################
    data = pd.concat(
        [
            vix["VIX_Close"],   # VIX 레벨
            kospi["KOSPI200"],  # KOSPI200 레벨
            fx["USD_KRW"],      # 환율 레벨
            target,             # VKOSPI 레벨
            DGS["DGS10"]        # 금리 레벨
        ],
        axis=1
    ).reset_index()  # Date 인덱스를 컬럼으로
    
    data.rename(columns={"Date": "orig_date", "VKOSPI": "target"}, inplace=True)
    return data

####################################### Metric #######################################
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    
def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


###################################### Loss Function ########################################
def MSE_loss(pred, target):
    return F.mse_loss(pred, target)

def MSE_diff_loss(pred, target):
    # (B, T, 1) -> (B, T)
    if pred.dim() == 3:
        pred = pred.squeeze(2)
        target = target.squeeze(2)

    pred_diff = pred[:, 1:] - pred[:, :-1] 
    target_diff = target[:, 1:] - target[:, :-1] 

    return F.mse_loss(pred_diff, target_diff)

def Trend_loss(pred, target, tau = 1.0):
    # (B, T, 1) -> (B, T)
    if pred.dim() == 3:
        pred = pred.squeeze(2)
        target = target.squeeze(2)

    pred_diff = pred[:, 1:] - pred[:, :-1] 
    target_diff = target[:, 1:] - target[:, :-1] 

    trend_label = (target_diff > 0).float()

    logits = pred_diff/tau

    return F.binary_cross_entropy_with_logits(logits, trend_label)