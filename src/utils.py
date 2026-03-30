import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return float(mean_absolute_error(y_true, y_pred))

def acc_at_k(y_true, y_pred, k=5):
    """Accuracy within k years"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= k))

def to_bin(ages, bins=[0,3,13,20,30,40,50,60,70,80,117], 
           labels=["0-2","3-12","13-19","20-29","30-39","40-49","50-59","60-69","70-79","80+"]):
    """Convert ages to age bins"""
    return pd.cut(pd.Series(ages), bins=bins, right=False, labels=labels).astype(str).values

def plot_confmat(y_true_bin, y_pred_bin, labels, title="Confusion matrix"):
    """Plot confusion matrix for age bins"""
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=labels)
    plt.figure(figsize=(8,6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    return cm

def compute_metrics_by_bin(df, true_col, pred_col, bins, labels):
    """Compute RMSE and MAE by age bins"""
    df = df.copy()
    df["abs_err"] = np.abs(df[true_col] - df[pred_col])
    df["sq_err"] = (df[true_col] - df[pred_col]) ** 2
    
    results = df.groupby("age_bin", observed=True).agg(
        n=(true_col, "size"),
        mae=("abs_err", "mean"),
        rmse=("sq_err", lambda x: float(np.sqrt(np.mean(x))))
    ).reindex(labels)
    
    return results
