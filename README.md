# VKOSPI Multivariate Forecasting Project

This repository contains a complete multivariate time-series forecasting study for the VKOSPI Index, using both classical statistical models and modern deep learning / foundation models.
The project is structured into two Jupyter Notebook files, each representing a different modeling paradigm.

<img width="973" height="508" alt="image" src="https://github.com/user-attachments/assets/51a7b187-1698-4236-8abd-ac7ff0bcbe44" />

## 📁 Project Structure
```
project_root/
│── vkospi_multivariate_forecasting.ipynb        # Preprocessing + AR/VAR/ADL + Granger causality
│── vkospi_multivariate_DNN_forecasting.ipynb    # Chronos-2 Zero-Shot + BiLSTM forecasting
│── data/
│     ├── VKOSPI_2004_2025_CL.csv
│     ├── DGS10_2004_2025.csv
│     └── ...
│── module.py
│── train_DL_batch.py
│── README.md
```

## 📌 1. vkospi_multivariate_forecasting.ipynb

This notebook contains all classical forecasting workflows, including preprocessing, statistical modeling, and causality analysis.

Included Sections

### ✓ Data Preprocessing
- Handling missing values
- Log-return construction
- Rate differencing
- Date alignment across all series

### ✓ AR (Autoregressive) Model
- BIC-based optimal lag selection
- Rolling 1-step-ahead forecasting
- RMSE / MAE / MAPE evaluation

### ✓ VAR (Vector Autoregression)
- Multivariate interaction modeling
- BIC lag selection
- Rolling-window forecasting for realistic performance evaluation

### ✓ ADL (Autoregressive Distributed Lag)
- Combines lagged target and lagged exogenous variables
- BIC-based selection for (p_y, p_x)
- Re-fitted at every test step

### ✓ Granger Causality Analysis
- Pairwise Granger tests
- p-value matrix and significance heatmap

### Outputs
- Full forecast curves
- Rolling predictions
- Error metric tables
- Causal relationship visualization

⸻

## 📌 2. vkospi_multivariate_DNN_forecasting.ipynb

This notebook includes deep learning and foundation-model-based forecasting methods.

⸻

### ✓ Chronos-2 Zero-Shot Forecasting

Amazon’s time-series foundation model.

Features:
- No training required
- Probabilistic forecasting (quantiles: 0.1, 0.5, 0.9)
- Iterative 1-step forecasting

Benefits:
- Strong out-of-the-box baseline
- Robust under volatility and regime shifts

⸻

### ✓ BiLSTM Forecasting

A lightweight learned model.

Model:
- Bidirectional LSTM
- Dense output head
- MSE loss with Adam optimizer

Training:
- Sliding-window batching
- Training script: train_DL_batch.py

Outputs:
- Full prediction curves
- Zoom-in comparisons
- RMSE / MAE / MAPE metrics

⸻

## 📊 Model Comparison Summary

Model	Category	Notes
AR	Statistical	Univariate baseline
VAR	Statistical	Captures multivariate dependencies
ADL	Statistical	Includes lagged exogenous variables
Chronos-2 Zero-Shot	Foundation Model	Strong no-training baseline
BiLSTM	Deep Learning	Learns nonlinear relationships


⸻

## 📈 Evaluation Metrics

All models are evaluated using the following metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

⸻

## 🚀 How to Run

### 1) Install dependencies

``` Requirments
pip install -r requirements.txt
```

### 2) Place datasets inside data/ directory

The VKOSPI and DGS10 datasets must be available.

### 3) Open notebooks

Statistical models:

- vkospi_multivariate_forecasting.ipynb

Deep-learning / foundation models:
How to Train
``` python
python train_DL_batch.py --model_name LSTM --batch_size 128 -- LR 1e-4 --max_epoch 400 --show_train_loss True --use_scaler True --ft_epoch 50 --ft_LR 1e-5 --window_len 300 --d_model 64
```
Check metrics and plotting
  
- vkospi_multivariate_DNN_forecasting.ipynb


⸻

## 🧾 Key Findings
	•	Chronos-2 provides strong zero-shot performance without training.
	•	BiLSTM can match or outperform classical models with sufficient training.
	•	VAR and ADL offer interpretability and strong short-term forecasting ability.
	•	AR serves as a simple but meaningful baseline.
	•	Granger causality reveals important directional effects between financial variables.

⸻

## 🙋 Contact

For questions, feedback, or collaboration inquiries, feel free to open an issue.

⸻
