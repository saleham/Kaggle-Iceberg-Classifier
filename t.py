import numpy as np
import pandas as pd
def load(train):
    # take the data to image format 75  * 75 and have a pre process on data by removing mean and var = 1
    ## as we see train['band_1'][i] is a list so we change its shape
    train_band_1 = np.array([np.array(flat).astype(np.float32) for flat in train["band_1"]])
    # train_band_1 = StandardScaler().fit_transform(train_band_1)
    train_band_1 = train_band_1.reshape(-1, 75, 75)
    train_band_2 = np.array([np.array(flat).astype(np.float32) for flat in train["band_2"]])
    # train_band_2 = StandardScaler().fit_transform(train_band_2)
    train_band_2 = train_band_2.reshape(-1, 75, 75)
    #train_dot = (train_band_1 - train_band_2) / 2
    train_avr_band = (train_band_1 + train_band_2) / 2
    ## get the whole input data to train_X
    train_X = np.concatenate([train_band_1[:, :, :, np.newaxis], train_band_2[:, :, :, np.newaxis],
                              train_avr_band[:, :, :, np.newaxis]], axis=-1)

    #####  set nan values in train and test data in angle column :)
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')  # NaN
    train['inc_angle'] = train['inc_angle'].fillna(0)
    X_angle = np.array(train['inc_angle'])
    X_angle = np.tile(X_angle[:, np.newaxis], (1, 2))

    return train_X,X_angle