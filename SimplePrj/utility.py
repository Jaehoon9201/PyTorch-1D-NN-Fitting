""" @ .py
  - utilities
 @author Jaehoon Shim
 @date 23.02.24
 @version 1.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import os



def data_split(data_csv):
    data_df = pd.read_csv(data_csv, index_col=0)
    x1 = data_df['x1'].to_numpy()
    x1 = np.expand_dims(x1, axis=1)
    x2 = data_df['x2'].to_numpy()
    x2 = np.expand_dims(x2, axis=1)
    x3 = data_df['x3'].to_numpy()
    x3 = np.expand_dims(x3, axis=1)
    x = np.append(x1, x2, axis=1)
    x = np.append(x, x3, axis=1)
    y = data_df['y'].to_numpy()

    return x, x1, x2, x3, y



def data_norm_forTrainData(norm_mode, x, x1, x2, x3, y):

    if norm_mode == 'WoNorm':
        x_norm0 = np.reshape(x[:, 0] , (1, -1))
        x_norm1 = np.reshape(x[:, 1] , (1, -1))
        x_norm2 = np.reshape(x[:, 2] , (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y )
        x1_norm = (x1)
        x2_norm = (x2 )
        x3_norm = (x3 )

    if norm_mode == 'Standard':
        y_tr_m = y.mean()
        y_tr_s = y.std()
        x1_tr_m = x[:, 0].mean()
        x1_tr_s = x[:, 0].std()
        x2_tr_m = x[:, 1].mean()
        x2_tr_s = x[:, 1].std()
        x3_tr_m = x[:, 2].mean()
        x3_tr_s = x[:, 2].std()
        if y_tr_s  < 0.001 : y_tr_s = 0.001
        if x1_tr_s < 0.001 : x1_tr_s = 0.001
        if x2_tr_s < 0.001 : x2_tr_s = 0.001
        if x3_tr_s < 0.001 : x3_tr_s = 0.001

        x_norm0 = np.reshape((x[:, 0] - x1_tr_m)/x1_tr_s, (1, -1))
        x_norm1 = np.reshape((x[:, 1] - x2_tr_m)/x2_tr_s, (1, -1))
        x_norm2 = np.reshape((x[:, 2] - x3_tr_m)/x3_tr_s, (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y - y_tr_m) / y_tr_s
        x1_norm  = (x1  - x1_tr_m) / x1_tr_s
        x2_norm = (x2 - x2_tr_m) / x2_tr_s
        x3_norm = (x3 - x3_tr_m) / x3_tr_s

    if norm_mode == 'MinMax':

        x_norm0 = np.reshape(  (x[:, 0] - np.min(x[:, 0])    )  /  (np.max(x[:, 0])- np.min(x[:, 0])  ), (1, -1))
        x_norm1 = np.reshape(  (x[:, 1] - np.min(x[:, 1])    )  /  (np.max(x[:, 1])- np.min(x[:, 1])  ), (1, -1))
        if prevent_NaN == 'ON' : x_norm2 = np.reshape(  (x[:, 2] - np.min(x[:, 2])    )  /  (np.max(x[:, 2])- np.min(x[:, 2])  ) + 0.001, (1, -1))
        if prevent_NaN == 'OFF': x_norm2 = np.reshape(  (x[:, 2] - np.min(x[:, 2])    ) /  (np.max(x[:, 2])- np.min(x[:, 2])  ) , (1, -1))
        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (  (y - np.min(y))      /  (np.max(y)- np.min(y))  )
        x1_norm      = (  (x1  - np.min(x[:, 0])     )      /  (np.max(x[:, 0])- np.min(x[:, 0]))    )
        x2_norm     = (  (x2 - np.min(x[:, 1])     )      /  (np.max(x[:, 1])- np.min(x[:, 1]))    )

    if norm_mode == 'MaxScaling':

        y_tr_max  = np.max(y)
        x1_tr_max = np.max(x[:, 0] )
        x2_tr_max = np.max(x[:, 1] )
        x3_tr_max = np.max(x[:, 2] )

        x_norm0 = np.reshape((x[:, 0] / x1_tr_max) , (1, -1))
        x_norm1 = np.reshape((x[:, 1] / x2_tr_max) , (1, -1))
        x_norm2 = np.reshape((x[:, 2] / x3_tr_max) , (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y / y_tr_max)
        x1_norm      = (x1  / x1_tr_max)
        x2_norm     = (x2 / x2_tr_max)
        x3_norm     = (x3 / x3_tr_max)


    return x_norm, x1_norm, x2_norm, x3_norm, y_norm



def data_norm_forTestData(norm_mode, x, x1, x2, x3, y,
                          optima_x, optima_y):

    if norm_mode == 'WoNorm':
        x_norm0 = np.reshape(x[:, 0] , (1, -1))
        x_norm1 = np.reshape(x[:, 1] , (1, -1))
        x_norm2 = np.reshape(x[:, 2] , (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y )
        x1_norm = (x1)
        x2_norm = (x2 )
        x3_norm = (x3 )

    if norm_mode == 'Standard':
        y_tr_m  = optima_y.mean()
        y_tr_s  = optima_y.std()
        x1_tr_m = optima_x[:, 0].mean()
        x1_tr_s = optima_x[:, 0].std()
        x2_tr_m = optima_x[:, 1].mean()
        x2_tr_s = optima_x[:, 1].std()
        x3_tr_m = optima_x[:, 2].mean()
        x3_tr_s = optima_x[:, 2].std()
        if y_tr_s  < 0.001 : y_tr_s = 0.001
        if x1_tr_s < 0.001 : x1_tr_s = 0.001
        if x2_tr_s < 0.001 : x2_tr_s = 0.001
        if x3_tr_s < 0.001 : x3_tr_s = 0.001

        x_norm0 = np.reshape((x[:, 0] - x1_tr_m)/x1_tr_s, (1, -1))
        x_norm1 = np.reshape((x[:, 1] - x2_tr_m)/x2_tr_s, (1, -1))
        x_norm2 = np.reshape((x[:, 2] - x3_tr_m)/x3_tr_s, (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y - y_tr_m) / y_tr_s
        x1_norm  = (x1  - x1_tr_m) / x1_tr_s
        x2_norm = (x2 - x2_tr_m) / x2_tr_s
        x3_norm = (x3 - x3_tr_m) / x3_tr_s

    if norm_mode == 'MinMax':

        x_norm0 = np.reshape(  (x[:, 0] - np.min(optima_x[:, 0])    )  /  (np.max(optima_x[:, 0])- np.min(optima_x[:, 0])  ), (1, -1))
        x_norm1 = np.reshape(  (x[:, 1] - np.min(optima_x[:, 1])    )  /  (np.max(optima_x[:, 1])- np.min(optima_x[:, 1])  ), (1, -1))
        if prevent_NaN == 'ON' : x_norm2 = np.reshape(  (x[:, 2] - np.min(optima_x[:, 2])    )  /  (np.max(optima_x[:, 2])- np.min(optima_x[:, 2])  ) + 0.001, (1, -1))
        if prevent_NaN == 'OFF': x_norm2 = np.reshape(  (x[:, 2] - np.min(optima_x[:, 2])    )  /  (np.max(optima_x[:, 2])- np.min(optima_x[:, 2])  ) , (1, -1))
        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        x1_norm      = x_norm0
        x2_norm     = x_norm1
        x3_norm     = x_norm2

        y_norm = (  (y - np.min(optima_y))      /  (np.max(optima_y)- np.min(optima_y))  )

    if norm_mode == 'MaxScaling':

        y_tr_max  = np.max(optima_y)
        x1_tr_max = np.max(optima_x[:, 0] )
        x2_tr_max = np.max(optima_x[:, 1] )
        x3_tr_max = np.max(optima_x[:, 2] )

        x_norm0 = np.reshape((x[:, 0] / x1_tr_max) , (1, -1))
        x_norm1 = np.reshape((x[:, 1] / x2_tr_max) , (1, -1))
        x_norm2 = np.reshape((x[:, 2] / x3_tr_max) , (1, -1))

        x_norm = np.concatenate([x_norm0, x_norm1])
        x_norm = np.concatenate([x_norm, x_norm2])
        x_norm = np.transpose(x_norm)

        y_norm = (y / y_tr_max)
        x1_norm      = (x1  / x1_tr_max)
        x2_norm     = (x2 / x2_tr_max)
        x3_norm     = (x3 / x3_tr_max)


    return x_norm, x1_norm, x2_norm, x3_norm, y_norm

def y_rescale(BeforeScaledy , yUsedInOptima , norm_mode_ ):
    if norm_mode_ == 'WoNorm'    :  y_model_opt_rescale = BeforeScaledy
    if norm_mode_ == 'MaxScaling':  y_model_opt_rescale = BeforeScaledy * np.max(yUsedInOptima)
    if norm_mode_ == 'Standard'  :  y_model_opt_rescale = BeforeScaledy * yUsedInOptima.std() + yUsedInOptima.mean()
    if norm_mode_ == 'MinMax'    :  y_model_opt_rescale = BeforeScaledy * (np.max(yUsedInOptima) - np.min(yUsedInOptima)) + np.min(yUsedInOptima)

    return y_model_opt_rescale
