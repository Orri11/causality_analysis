import numpy as np
import pandas as pd

# Metrics
def RSE(pred, true):
    """
    Calculates relative quared error.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    """
    Calculates correlation coefficient.
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Calculates mean absolute error.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculates mean squared error.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculates root mean suared error.
    """
    return np.sqrt(MSE(pred, true))


def SMAPE(pred, true):
    """
    Calculates mean absolute percentage error.
    """
    return np.mean(2* np.abs(pred - true) / ( np.abs(pred) + np.abs(true) ) )

def mase_greybox(holdout, forecast, scale):
    """
    Calculates Mean Absolute Scaled Error as in Hyndman & Koehler, 2006.
    
    Reference: https://github.com/config-i1/greybox/blob/6c84c729786f33a474ef833a13b7715831bd29e6/R/error-measures.R#L267

    Parameters:
        holdout (list or numpy array): Holdout values.
        forecast (list or numpy array): Forecasted values.
        scale (float): The measure to scale errors with. Usually - MAE of in-sample.
        na_rm (bool, optional): Whether to remove NA values from calculations.
                                Default is True.

    Returns:
        float: Mean Absolute Scaled Error.
    """
    if len(holdout) != len(forecast):
        print("The length of the provided data differs.")
        print(f"Length of holdout: {len(holdout)}")
        print(f"Length of forecast: {len(forecast)}")
        raise ValueError("Cannot proceed.")
    else:
        return np.mean(np.abs(np.array(holdout) - np.array(forecast)) / scale)


def MASE(pred, true, df_a, seasonality_period):
    """
    Calculates mean squared percentage error.
    """
    mase_vector = []
    for i in range (len(df_a.columns)):
        lagged_diff = [df_a.iloc[j,i] - \
                       df_a.iloc[j - seasonality_period,i] for j in range(seasonality_period, len(df_a))]
        mase_vector.append(mase_greybox(true, pred, np.mean(np.abs(lagged_diff))))
    mean_mase = np.mean(mase_vector)
    return mean_mase

def metric(pred, true, df_a, seasonality_period):
    """
    Wraps up metric functions, calculates and returns all.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    smape = SMAPE(pred, true)
    mase = MASE(pred, true, df_a, seasonality_period)

    return mae, mse, rmse, smape, mase