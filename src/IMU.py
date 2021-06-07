# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:07:43 2021

@author: andhaugl
"""
import numpy as np
from quaternion import *

# %% IMU Implementation
"""
Implement IMU here
Gyroscope measures angular rates 
Accelerometer measures specific forces

"""
def z_acc(
        euler_angles: np.ndarray,
        g_nb_n: np.ndarray,
        acc_nb_n: np.ndarray,
        accel_bias_b: np.ndarray,
       # accel_noise_b: np.ndarray
       )-> np.ndarray:
    """
    # Equation 14.16 Fossen Draft 2021, bias is modeled after 14.17
    Parameters
    ------
    euler_angles: pitch, roll and yaw
    quaternions: Attitude quaternions
    g_nb_n: True gravity vector in NED (3,)
    acc_nb_n: True acceleration vector in NED (3,)
    accel_bias_b: Accelerometer bias in Body (3,)
    accel_noise_b: Accelerometer noise in Body (3,) Generated random noise, noise_dot = additive gaussian noise

    Returns
    # -------
    Returns specific forces in body, [z_acc_b], Rounded to only have 7 numbers

    """
    #np.random.normal(mean, std)
    acc_noise_b = np.array([np.random.normal(0, 0.001),
                    np.random.normal(0, 0.001),
                    np.random.normal(0, 0.001)])
    acc_noise_b

    
    # R = rot_mat(euler_angles) #Can also be performed with quat to rotation
    # R = quaternion_to_rotation_matrix(quaternions)
    # z_acc_b = R.T @ (acc_nb_n.T - g_nb_n).T + accel_bias_b + accel_noise_b #deg/s
    z_acc_b = (acc_nb_n - g_nb_n) + accel_bias_b + acc_noise_b #deg/s
    z_acc_b = np.round(z_acc_b,7)
    
    assert z_acc_b.shape ==(
        3,), f"accel_meas: acceleration vector shape incorrect. Expected (3,), got {z_acc_b.shape}"
    
    return z_acc_b

def z_gyro(
              omega_t_b: float,
              omega_bt_b: float,
             # omega_n_b: float
        ): 
    """
    #Equation 14.6 and 14.7 in Fossen Draft 2021

    Parameters
    ----------
     omega_t: True angular velocity
     omega_bt: gyro bias
     omega_n: gyro noise
     
    Returns
    -------
    Measured angular rates in XYZ [rad/s]. Rounded to only have 7 numbers

    """
    
    omega_noise_b = np.array([np.random.normal(0, 0.00004),
                    np.random.normal(0, 0.00004),
                    np.random.normal(0, 0.00004)])

    z_gyro_b = (omega_t_b + omega_bt_b + omega_noise_b)
    z_gyro_b = np.round(z_gyro_b,7)
    
    assert z_gyro_b.shape ==(
        3,), f"z_gyro: angular rate vector shape incorrect. Expected (3,), got {z_gyro_b.shape}"
    
    return z_gyro_b