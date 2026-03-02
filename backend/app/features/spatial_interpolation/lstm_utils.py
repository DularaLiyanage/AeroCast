# lstm_utils.py
import numpy as np
import pandas as pd

def build_lstm_sequence(
    df_history: pd.DataFrame,
    current_row: dict,
    feature_cols: list,
    window: int = 24
):
    """
    Build (1, window, features) tensor for LSTM inference.

    df_history : past data BEFORE time T (filtered by station)
    current_row: current input row at time T
    """

    hist = df_history.tail(window - 1)

    if len(hist) < window - 1:
        raise ValueError("Not enough historical data for LSTM sequence")

    hist_values = hist[feature_cols].values

    current_values = np.array(
        [current_row[c] for c in feature_cols],
        dtype=float
    ).reshape(1, -1)

    seq = np.vstack([hist_values, current_values])
    return seq.reshape(1, window, len(feature_cols))
