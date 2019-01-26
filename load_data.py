import numpy as np
import pandas as pd
def load(train):
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    train_X = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                              ((X_band_1 + X_band_2) / 2)[:, :, :, np.newaxis]], axis=-1)

    #####  set nan values in train and test data in angle column :)
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')  # NaN
    train['inc_angle'] = train['inc_angle'].fillna(0)
    X_angle = np.array(train['inc_angle'])
    X_angle = np.tile(X_angle[:, np.newaxis], (1, 2))

    return train_X,X_angle