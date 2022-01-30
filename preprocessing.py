import numpy as np
import pandas as pd

WINDOWS = [10, 50]

def create_rolling_features(df):
    for window in WINDOWS:
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] - df["rolling_min_" + str(window)]

    df = df.replace([np.inf, -np.inf], np.nan)    
    df.fillna(0, inplace=True)
    return df


def create_features(df, batch_size):
    
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    for window in WINDOWS:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
        
    df['signal_2'] = df['signal'] ** 2
    return df   

# reading data
test  = pd.read_csv(f'test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
y_test_proba = np.load(f"Y_test_proba.npy")

for i in range(11):
    test[f"proba_{i}"] = y_test_proba[:, i]
    
train_mean = 0.08263289928436279 
train_std = 2.4800126552581787
test = create_rolling_features(test)   
test['signal'] = (test.signal - train_mean) / train_std

test = create_features(test, 4000)

cols_to_remove = ['time','signal','batch','batch_index','batch_slices','batch_slices2', 'group']
cols_test = [c for c in test.columns if c not in cols_to_remove]
X_test = test[cols_test]