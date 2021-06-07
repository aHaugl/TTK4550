# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:42:48 2021

@author: andhaugl
Simulation with ideal imu. Predict velocity, position and attitude from gyro and accelerometer
measurements without noise and bias through a Kalman prediction step. Attitude representation
is found through both euler angles and quaternions.
"""

# %% 
from plotter import plot_NEES, plot_NIS, plot_angle, plot_error_pos_sigma, plot_error_vel_sigma, plot_path, state_error_plots
from typing import Tuple, Sequence, Any
from matplotlib import pyplot as plt

# from plotfunctionstesting import plot_acc, plot_acc2, plot_gyro, plot_angle

import numpy as np
import scipy.linalg as la
import scipy
import scipy.io
import scipy.stats
from dataclasses import dataclass, field
import timeit
import time

import tqdm
import os
import sys

from tqdm import trange
from tqdm import tqdm_notebook

from quaternion import * 
from cat_slice import CatSlice
# from eskf import ESKF
# from eskf import ESKF
from eskf_pseudoranges import ESKF
from eskf_runner import run_eskf
from plotter import * #plot_error_v_sigma, plot_pos, plot_vel, plot_angle, plot_estimate, plot_3Dpath, plot_path, state_error_plots, plot_NEES, plot_NIS
# from timer import * 

# %% plot config check and style setup

plt.close('all')

from plot_setup import setup_plot
setup_plot()
np.seterr(all='raise')
scipy.special.seterr(all='raise')

# %% Q and R matrixes are defined as
"""
Q_err: acc_std, gyro_std, acc_bias_Std, gyro_bias std
R: p_std -> R_GNSS = p_std**2*eye(3)
"""

# %% Load data and plot 
"""
Loads true state, time steps and other simulation variables. Run gen_mat to generate
new data
"""

folder = os.path.dirname(__file__)
# filename_to_load = f"{folder}/../data/simulation_params.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver_long.mat"
filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver_long_ver2.mat"
cache_folder = os.path.join(folder,'..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

timeIMU = loaded_data["timeIMU"].ravel()
#acc_t = loaded_data["acc_t"].T
if 'x_true' in loaded_data:
    x_true = loaded_data["x_true"].T
else:
    x_true = None
# z_GNSS = loaded_data["z_GNSS"].T
    
beacon_location = loaded_data["beacon_location"]

dt = np.mean(np.diff(timeIMU))

# %% indices and initialization
POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)

ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)

# %% 

# Initializing prediction
x_pred_init = np.zeros(16)

# 8 figure init
x_pred_init[POS_IDX] = np.array([10,0,1])
x_pred_init[VEL_IDX] = np.array([0, 2, 0])
x_pred_init[6:10] = np.array([1,0,0,0]) #(0,0,0) iny euler
# x_pred_init[10:13] = np.array([-8.75e-2,-8.74e-2,6.9e-2])
# x_pred_init[13:16] = np.array([-9.54e-4,-7.32e-04,3.37e-04])

# x_pred_init[POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
# x_pred_init[VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
# no initial rotation: nose to North, right to East, and belly down
# x_pred_init[6] = 1


# %% From another simulation

# %% Q matrix params
# acc_std = 7.59562072e-02
# rate_std =  5.12679127e-02

# cont_acc_bias_driving_noise_std =  1.16562328e-04
# cont_rate_bias_driving_noise_std =  1.94631260e-05


# %% 
# %% Current best
# TUNABLE0.01035
# cont_gyro_noise_std =  0.01035
cont_gyro_noise_std =   4.36e-5  # (rad/s)/sqrt(Hz)
# TUNABLE 0.015191 
# cont_acc_noise_std = np.array([0.01, 0.01, 0.042191])
# cont_acc_noise_std = 0.01549 #1.167e-2  # (m/s**2)/sqrt(Hz)
cont_acc_noise_std =  1.167e-2  # (m/s**2)/sqrt(Hz)

# cont_gyro_noise_std = 0.001      #4.36e-5  # (rad/s)/sqrt(Hz)
# cont_acc_noise_std = 0.001  #1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
# This is the formula from example 4.9. Should this be the driving noise instead?
rate_std = np.round(0.5 * cont_gyro_noise_std * np.sqrt(1 / dt),7) #Just to avoid float shit
acc_std = np.round(0.5 * cont_acc_noise_std * np.sqrt(1 / dt),7)

# Bias values
# TUNABLE 0.0005839  0.00005839 
# 5e-5 From another simulation
rate_bias_driving_noise_std =  5e-5 

cont_rate_bias_driving_noise_std = (
    np.round((1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt), 7)
)
#4e-3 From another simulation
#0.0001947, 0.0020947 0.0050947
acc_bias_driving_noise_std = 4e-3
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)

# %% Used to approx acc bias and gyro bias >0
p_acc =   1.000e-6
p_gyro = 1.00000000e-6

# %% Position and velocity measurement, Used in R
# p_std = np.array([1.58170828, 1.58170828, 0.97243219])   # Measurement noise

#The measurements from the GNSS are perfect at the moment
# p_std = np.array([0, 0, 0])
p_std = np.array([.01, .01, .03])
# p_std = np.array([.1, .1, .3])
# p_std = np.array([.3, .3, .5])

# %% Cov matrix initalization

P_pred_init = np.zeros((15,15))
P_pred_init[POS_IDX ** 2] = np.eye(3) * 3**2  
P_pred_init[VEL_IDX ** 2] = np.eye(3) * 2**2  #0.2
P_pred_init[ERR_ATT_IDX ** 2] = np.eye(3) * 0.5**2  # TODO
P_pred_init[ERR_ACC_BIAS_IDX ** 2] = np.eye(3) * 0.01**2 
P_pred_init[ERR_GYRO_BIAS_IDX ** 2] = np.eye(3) * 0.001**2 # 0.001**2 

 
eskf_parameters = [acc_std,
                    rate_std,
                    cont_acc_bias_driving_noise_std,
                    cont_rate_bias_driving_noise_std,
                    p_acc,
                    p_gyro]   

# P_pred_init_list = [P_pred_init_pos,
#                     P_pred_init_vel,
#                     P_pred_init_err_att,
#                     P_pred_init_err_acc_bias,
#                     P_pred_init_err_gyro_bias]

#init_parameters = [x_pred_init, P_pred_init]

# %% Run estimation for
#Number of seconds of simulation to run. len(timeIMU) decides max
# N: int = int(6000/dt)
N: int = int(600/dt) 
# N: int = int(30000) 
# N: int = int(90000)
# N: int = len(timeIMU)
offset = 0
doGNSS: bool = True


# TODO: Set this to False if you want to check that the predictions make sense over reasonable time lenghts

#parameters = eskf_parameters + init_parameters

# %% Run eskf_runner  
"""
Running the simulation
"""
beacon_location: np.ndarray = loaded_data["beacon_location"]

use_batch_pseudoranges: bool = True
use_iterative_pseudoranges: bool = False

# num_beacons = len(beacon_location)
num_beacons = 4
num_sims = 20

t_batch = np.zeros(num_sims)
elapsed_batch = np.zeros(num_sims)

print("Number of beacons used: ", num_beacons)
print("Number of simulations ran through", num_sims)
print("Simulation duration (seconds): ", N*dt) 

if (use_batch_pseudoranges):
    print("Using batch pseudoranges")
elif(use_iterative_pseudoranges):
    print("Using iterative pseudoranges")

for i in range(num_sims):  
    # timeit.timeit()
    
    t_batch[i] = time.time()
    
    (x_pred,
    x_est,
    P_est,
    GNSSk,
    # toc
    ) = run_eskf (N, loaded_data,
                        eskf_parameters,
                        x_pred_init, P_pred_init, p_std, 
                        num_beacons,
                        use_batch_pseudoranges,
                        use_iterative_pseudoranges,
                        offset =0.0, 
                        use_GNSSaccuracy=False, doGNSS=True,
                        debug=False  )
          
    elapsed_batch[i] = time.time() - t_batch[i] 
    
print("Ellapsed time: ", elapsed_batch)

average_time_batch = np.average(elapsed_batch)
max_time_batch = np.amax(elapsed_batch)
min_time_batch = np.amin(elapsed_batch)


print("Average time elapsed: ", average_time_batch, "seconds")
print("Max_time_batch elapsed: ", max_time_batch, "seconds")
print("min_time_batch elapsed: ", min_time_batch, "seconds")


# %% 
use_batch_pseudoranges: bool = False
use_iterative_pseudoranges: bool = True

t_iterative = np.zeros(num_sims)
elapsed_iterative = np.zeros(num_sims)

print("Number of beacons used: ", num_beacons)
print("Number of simulations ran through", num_sims)
print("Simulation duration (seconds): ", N*dt) 

if (use_batch_pseudoranges):
    print("Using batch pseudoranges")
elif(use_iterative_pseudoranges):
    print("Using iterative pseudoranges")

for i in range(num_sims):  
    # timeit.timeit()
    
    t_iterative[i] = time.time()
    (x_pred,
    x_est,
    P_est,
    GNSSk,
    # toc
    ) = run_eskf (N, loaded_data,
                        eskf_parameters,
                        x_pred_init, P_pred_init, p_std, 
                        num_beacons,
                        use_batch_pseudoranges,
                        use_iterative_pseudoranges,
                        offset =0.0, 
                        use_GNSSaccuracy=False, doGNSS=True,
                        debug=False  )
          
    elapsed_iterative[i] = time.time() - t_iterative[i] 
    
print("Ellapsed time: ", elapsed_iterative)

average_time_iterative = np.average(elapsed_iterative)
max_time_iterative = np.amax(elapsed_iterative)
min_time_iterative = np.amin(elapsed_iterative)

print("Average time iterative elapsed: ", average_time_iterative, "seconds")
print("Max_time_iterative elapsed: ", max_time_iterative, "seconds")
print("Min_time_iterative elapsed: ", min_time_iterative, "seconds")

print("Average time iterative elapsed: ", average_time_iterative, "seconds")

speed_up = (average_time_batch - average_time_iterative)/average_time_batch*100
print("Speed up time: ", speed_up, "secons")
# %% Plots and stuff                           

# plt.close("all")
t = np.linspace(0,dt * (N-1), N)
# plot_path(t, N, pos_t, pos_t)
tGNSS = loaded_data["timeGNSS"].T
z_GNSS = loaded_data["z_GNSS"].T

z_acc_vector = loaded_data["z_acc"].T
acc_t = loaded_data["acc_t"].T
omega_t = loaded_data["omega_t"].T
z_gyro_vector = loaded_data["z_gyro"].T


# # %% Estimation plots

# plot_pos(t, N, x_est, tGNSS, GNSSk, z_GNSS, x_true)
# plot_vel(t, N, x_est, x_true)
# plot_acc(t, N, z_acc_vector, acc_t)
# # %%
# # plot_gyro(t, N, z_gyro_vector, omega_t)
# # %%
# plot_angle(t, N, x_est, x_true)
# # 
# # plot_estimate(t, N, x_est)

# plot_path(t,N, beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
# # # %%
# plot_3Dpath(t, N,beacon_location[:num_beacons], GNSSk, z_GNSS, x_est, x_true)
# # %%

# state_error_plots(t, N, x_est, x_true, delta_x)

# %% Validation plots
# plot_error_pos_sigma(x_est, x_true, P_est, N)
# plot_error_vel_sigma(x_est, x_true, P_est, N)
# plot_error_att_sigma(x_est, x_true, P_est, N)
# plot_error_acc_bias_sigma(x_est, x_true, P_est, N)
# plot_error_rate_bias_sigma(x_est, x_true, P_est, N)

# # error_distance_plot(t, N, dt, GNSSk, x_true, delta_x, z_GNSS)
# # %% 
# plot_NIS(NIS)
# plot_NEES(t, N, dt,
#           NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias, NEES_gyrobias,
#             confprob=0.95)
