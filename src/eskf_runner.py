# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:39:06 2021

@author: Andreas
"""
import numpy as np
from tqdm import trange
from tqdm import tqdm_notebook
from timer import * 

from eskf_pseudoranges import (
# from eskf import (    
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from IMU import z_acc, z_gyro


def run_eskf(N, loaded_data,
              eskf_parameters,
              x_pred_init, P_pred_init, p_std,
              num_beacons,
              use_batch_pseudoranges,
              use_iterative_pseudoranges,
              offset =0.,
              use_GNSSaccuracy=False, doGNSS=False,
              debug=False ):
    
# def run_eskf(eskf_parameters, x_pred_init, P_pred_init_list, loaded_data,
#              p_std, N, use_GNSSaccuracy=False, doGNSS=True,
#              debug=False, offset=0.):
    """
    Description:
        Unravels, initializes and runs error state kalman filter 
        for parameters set in .mat file and main

    Parameters
    ----------
    N :  Number of steps to run
    loaded_data : Loaded data matrix
    eskf_parameters : 
    x_pred_init : State prediction initialization params
    P_pred_init_list : Covariance init params
    use_GNSSaccuracy : TYPE, optional
        DESCRIPTION. The default is False.
    doGNSS : TYPE, optional
        DESCRIPTION. The default is False.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    offset : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    result : List of returns from ESKF

    """
    
    # %% Read loaded data
    if "x_true" in loaded_data:
        x_true = loaded_data["x_true"].T
    else:
        x_true = None
    z_acc = loaded_data["z_acc"].T
    z_gyro = loaded_data["z_gyro"].T
    z_GNSS = loaded_data["z_GNSS"].T
#    acc_t = loaded_data["acc_t"].T
    
    if use_GNSSaccuracy:
        GNSSaccuracy = loaded_data['GNSSaccuracy'].T
    else:
        GNSSaccuracy = None
    
    # S_a: np.ndarray = np.eye(3)
    # S_g: np.ndarray = np.eye(3)
    
    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]
    
    # %% Beacon stuff
    # The matrix has 15 beacons implemented
    beacon_location: np.ndarray = loaded_data["beacon_location"]
    #To select x amount of beacons, use beacon_location = beacon_location[:x], x =[0,15]
    beacon_location = beacon_location[:num_beacons]
    R_beacons = np.zeros((num_beacons,num_beacons))
    np.fill_diagonal(R_beacons, p_std**2)
    # np.fill_diagonal(R_beacons,np.array([1,1,1])**2)
    
    # %% 
    lever_arm = loaded_data["leverarm"].ravel()
    
    timeGNSS = loaded_data["timeGNSS"].ravel()
    timeIMU = loaded_data["timeIMU"].ravel()
    Ts_IMU = [0, *np.diff(timeIMU)]
    
    steps = len(z_acc)
    gnss_steps = len(z_GNSS)
        
    # %% Initialize state predictions, estimates and NEES   
    x_pred: np.ndarray = np.zeros((steps, 16))
    x_pred[0] = x_pred_init

    P_pred = np.zeros((N, 15, 15))
    P_pred[0] = P_pred_init
    
        #Initialize the kalman filter
    eskf = ESKF(
        *eskf_parameters,
        S_a=S_a,  # set the accelerometer correction matrix
        S_g=S_g,  # set the gyro correction matrix,
        debug=debug
    )
    R_GNSS = np.diag(p_std ** 2)
    
    x_est: np.ndarray  = np.zeros((N, 16))
    P_est = np.zeros((N, 15, 15))
  
     # keep track of current step in GNSS measurements
    offset += timeIMU[0]
    GNSSk_init = np.searchsorted(timeGNSS, offset)
    GNSSk = GNSSk_init
    offset_idx = np.searchsorted(timeIMU, offset)
    timeIMU = timeIMU[offset_idx:]
    z_acc = z_acc[offset_idx:]
    z_gyro = z_gyro[offset_idx:]
    Ts_IMU = Ts_IMU[offset_idx:]
    k = 0


    # %% 
    # print("Starting timer")
    # tic()
    for k in trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            if use_GNSSaccuracy:
                R_GNSS_scaled = R_GNSS * GNSSaccuracy[GNSSk]
            else:
                R_GNSS_scaled = R_GNSS


            # %% Test if range and LOS_matrix works
            
            x_est[k], P_est[k] = eskf.update_GNSS_position(x_pred[k],
                                                           P_pred[k],
                                                           z_GNSS[GNSSk],
                                                           R_GNSS_scaled,
                                                           R_beacons,
                                                           beacon_location,
                                                           use_batch_pseudoranges,
                                                           use_iterative_pseudoranges,
                                                           lever_arm
                                                           )

            #Ranges
            assert np.all(np.isfinite(P_est[k])
                          ), f"Not finite P_pred at index {k}"

            GNSSk += 1
        else:
            # No updates, est = pred
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]

            
        if k < N - 1:
            x_pred[k+1], P_pred[k+1] = eskf.predict(
                                            x_est[k],
                                            P_est[k],
                                            # acc_t[k],
                                            z_acc[k], #Denne er k pga måten dataen er laget på (?)
                                            z_gyro[k+1],
                                            Ts_IMU[k]
                                             )

    result = (
              x_pred,
              x_est,
              P_est,
              GNSSk,     
              )


    return result