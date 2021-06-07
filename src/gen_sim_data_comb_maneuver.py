# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:27:21 2021

@author: Andreas
"""

import numpy as np
import scipy.io
from tqdm import trange

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__)) #test
parentdir = os.path.dirname(currentdir)  #src
sys.path.append(parentdir)

from cat_slice import CatSlice
from IMU import z_acc, z_gyro
from quaternion import transf_mat, euler_to_quaternion

POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)

# %% 

Ts = 1/100 # Sampling rate [Hz]
Ts_GNSS = 1 #Sampling rate [Hz]
time_horizon = 60*60*1 #[seconds]
timeIMU = np.arange(0, time_horizon, Ts)
timeGNSS = np.arange(0.990,time_horizon, Ts_GNSS)

N = len(timeIMU)
M = len(timeGNSS)

dt = np.mean(np.diff(timeIMU))
t = np.linspace(0,dt * (N-1), N)
x_true: np.ndarray = np.zeros((len(timeIMU), 16)) # State_vector
# x_true = x_true.astype('np.double')

# %% Load data and plot 
"""
Loads true state, time steps and other simulation variables. Run gen_mat to generate
new data
"""

folder = os.path.dirname(__file__)
filename_to_load = f"{folder}/../data/task_simulation.mat"
# filename_to_load = f"{folder}/../data/simulation_params_comb_maneuver.mat"
cache_folder = os.path.join(folder,'..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

temp = loaded_data["xtrue"].T



# %% # %% NED Position 
r = 10
omega = 1/10
z_ampl = 1

# %% Attitude
euler_angles: np.ndarray = np.zeros(3)
eul_att: np.ndarray = np.zeros((N+1,3))
quat_t: np.ndarray = np.zeros((N,4))
x_true[0,ATT_IDX] = np.array([1,0,0,0])

rollVelTrue = np.zeros((N+1, 3)) #rad/s

z_acc_vector: np.ndarray = np.zeros((N+1,3))#m/s^2
z_acc_vector[0] = np.array([0  , 0.   , 0])

z_gyro_vector: np.ndarray = np.zeros((N, 3))
z_gyro_vector[0,:] = np.array([0,0,0]) #rad/s

# %% Biases
a_bt = np.zeros(3)
# a_bt = np.array([1,1,1,])
omega_bt = np.zeros(3)
# 
# %% Noises

a_n: np.ndarray = np.zeros(3)
omega_n = np.zeros(3)

g_t: np.ndarray = np.array([0, 0, 9.82 ])

# %% NED Acceleration: v_dot = a

acc_t: np.ndarray = np.zeros((N,3))
# acc_t = acc_t.astype('float32')

# %%  GNSS measurements
z_GNSS = np.zeros((M,3))
# %% Kalman prediction testing
vel_pred_test = np.zeros((N+1,3))
pos_pred_test = np.zeros((N+1,3))

vel_pred_test[0] = np.array([0, 2.0, 0])
pos_pred_test[0] = np.array([10,0,1])
t = 0
rot_t = 0
tmc = 20 #time for maneuver change

# %% 
# x_true[:,ACC_BIAS_IDX] = temp[:,ACC_BIAS_IDX]
# x_true[:,GYRO_BIAS_IDX] = temp[:,GYRO_BIAS_IDX]

print("Generating GNSS-ranges without noise")
for i in trange(M): #M = len(timeGNSS)
    # Generate range for first 20 seconds (no noise)    
    if timeGNSS[i] <= tmc:
        z_GNSS[i, 0] = 10
        z_GNSS[i, 1] = 2*timeGNSS[i]
        z_GNSS[i, 2] = 1 
        
    if timeGNSS[i] > tmc:
        
        z_GNSS[i, 0] = (r * np.cos(omega * timeGNSS[i-tmc]) 
                        )
        
        z_GNSS[i, 1] = (1/10 *(r ** 2) * np.sin(2 * omega * timeGNSS[i-tmc])
                        + 40
                        )
        
        z_GNSS[i, 2] = (1/z_ampl * r * omega * np.cos((omega / z_ampl ) * timeGNSS[i-tmc]))
        
        
print("Generating true POS, VEL, ACC, ACC_BIAS, GYRO_BIAS and gyro/acc-measurements. \n")
for k in trange(N): #N = len(timeIMU)

    x_true[k, ACC_BIAS_IDX] = np.array([-0.00875,
                                        -0.00875,
                                        0.00690])
    
    x_true[k, GYRO_BIAS_IDX] = np.array([-9.5e-4 ,
                                          -7.3e-4,
                                          3.7e-4])
    
    
    # # #Accelerometer bias is typically measured in milli-g's (m/s^2)
    
    # #Skal det modelleres som dette? Dette er det som er brukt i den lange matrisen?
    # if k > 0:
    #     x_true[k, ACC_BIAS_IDX] =  x_true[k-1, ACC_BIAS_IDX] + np.array(
    #         [np.random.normal(0, 0.001),
    #          np.random.normal(0, 0.001),
    #          np.random.normal(0, 0.001)]
    #         )
        
    #     x_true[k, GYRO_BIAS_IDX] = x_true[k-1, GYRO_BIAS_IDX] + np.array(
    #             [np.random.normal(0, 0.000001),
    #              np.random.normal(0, 0.000001),
    #              np.random.normal(0, 0.000001)]
    #             )
    
    #Accelerometer bias is typically measured in milli-g's (m/s^2)
    if k > 0:
        x_true[k, ACC_BIAS_IDX] =  np.array([-0.00875 + np.random.normal(0, 0.004),
                                              -0.00875 + np.random.normal(0, 0.004),
                                              0.00690 + np.random.normal(0, 0.004)])
        
        x_true[k, GYRO_BIAS_IDX] = np.array([-9.5e-4 + np.random.normal(0, 0.00005),
                                              -7.3e-4 + np.random.normal(0, 0.00005),
                                              3.7e-4 + np.random.normal(0, 0.00005)])



    #Move east for 20 seconds with 2m/s
    if timeIMU[k] <= tmc:
        x_true[k, 0] = 10
        x_true[k, 1] = 2*timeIMU[k]
        x_true[k, 2] = 1
        
        # %%  # %% NED Velocity p_dot = v
        x_true[k, 3] = 0 #Constant value
        x_true[k, 4] = 2
        x_true[k, 5] = 0
        
        acc_t[k,0] = 0
        acc_t[k,1] = 0
        acc_t[k,2] = 0
        
                
        
    # %% Initialize path for 8 curve
    # if timeIMU[k] == 20:
    #     x_true[k, 0] = 10
    #     x_true[k, 1] = 40
    #     x_true[k, 2] = 1
        
    #     x_true[k, 3] = 0 #Constant value
    #     x_true[k, 4] = 2
    #     x_true[k, 5] = 0
        
    #     acc_t[k, 0] = -0.1
    #     acc_t[k, 1] = 0
    #     acc_t[k, 2] = -0.01
        
    # %% Start 8 curve at [10,40,1]
    if timeIMU[k] > tmc: 
        # x_true[k, 0] = 1*timeIMU[k]-60
        # x_true[k, 1] = x_true[k-1,1]
        # x_true[k, 2] = 0*timeIMU[k]

        x_true[k, 0] = (r * np.cos(omega * timeIMU[t]) 
                        )
        
        x_true[k, 1] = (1/10 *(r ** 2) * np.sin(2 * omega * timeIMU[t])
                        + 40
                        )
        
        x_true[k, 2] = (1/z_ampl * r * omega * np.cos((omega / z_ampl ) * timeIMU[t])
                        )
        
            # Velocity       
        x_true[k, 3] = (- omega * r * np.sin(omega * timeIMU[t])
                        )
        
        x_true[k, 4] = (1/5 * omega * (r ** 2) *np.cos(2 * omega * timeIMU[t] )
                      )
        
        x_true[k, 5] = (-1/(z_ampl**2) * r * (omega **2) * np.sin((omega / z_ampl ) * timeIMU[t] )
                         )

        acc_t[k,0] = (- (omega ** 2) * r * np.cos(omega * timeIMU[t])
                     )
        
        acc_t[k,1] = (- 2/5 * (omega ** 2) * (r**2) * np.sin(2* omega * timeIMU[t])
                      )
        
        acc_t[k,2] = (-1/(z_ampl**3) * (omega ** 3) * r *np.cos((omega / z_ampl ) * timeIMU[t])
                     )
        t += 1
        
    vel_pred_test[k+1] = vel_pred_test[k] + Ts * (acc_t[k])
    pos_pred_test[k+1] = pos_pred_test[k] + Ts * vel_pred_test[k] + (Ts **2)/2 * (acc_t[k])
    
    # %% Gyro rate input
    
    #over 10 seconds, pitch 45 degrees (np.pi/4 = 0.7853981633974483 rad)
    if (timeIMU[k] >= 10) and (timeIMU[k] <100 ):
        rollVelTrue[k,:] = np.round(np.array([4.5,0,0])*np.pi/180,7) #rad/s
        # print("i >5000")
        
    #over 10 seconds, roll 45 degrees (np.pi/4 = 0.7853981633974483 rad)   
    if (timeIMU[k] >= 100) and (timeIMU[k] <200 ):
        # print(i>12500)
        rollVelTrue[k,:] = np.round(np.array([0,4.5,0])*np.pi/180,7)
        
    #over 10 seconds, yaw  45 degrees (np.pi/4 = 0.7853981633974483 rad)
    if (timeIMU[k] >= 200) and (timeIMU[k] < 300 ):
        rollVelTrue[k,:] = np.round(np.array([0, 0 , 4.5])*np.pi/180,7)
        
    if (timeIMU[k] >= 300):
        rollVelTrue[k,:] = np.round(np.array([5*np.sin(1/10000 *rot_t),
                                              5*np.sin(1/10000 *rot_t),
                                              5*np.sin(1/10000 *rot_t)])*np.pi/180,7)
        rot_t +=1
    if (i> 20000):
        print(rollVelTrue[:,i])

    
    
    # %% Gyroscope #Outputs in rad/s since input is in rad/s
    z_gyro_vector[k,:]  = z_gyro(rollVelTrue[k,:],
                                     x_true[k,GYRO_BIAS_IDX],
                                    # omega_n
                                     )
    #Eq 2.39 Fossen 21
    eul_att[k+1,:] = eul_att[k,:] + transf_mat(eul_att[k,:]) @ z_gyro_vector[k,:]*Ts

     # %% Accelerometer
    z_acc_vector[k,:] = z_acc(eul_att[k],
                                g_t,
                                acc_t[k,:],
                                x_true[k, ACC_BIAS_IDX],
                                #a_n
                                )
# %% 
print("Rounding and adding measurements to a .mat file. \n")
eul_att = eul_att[:N,:] 
# z_acc_vector = z_acc_vector[:N+1,:] #Dette forskyver feilen ein indeks tidligere
# z_acc_vector = z_acc_vector[:N,:]
quat_t = np.zeros((N,4))
quat_t = np.apply_along_axis(euler_to_quaternion,1,eul_att)
# %% 
x_true[:, ATT_IDX] = quat_t

x_true = np.apply_along_axis(np.round,1 ,x_true,7)
acc_t = np.apply_along_axis(np.round,1 ,acc_t,7)
vel_pred_test = np.apply_along_axis(np.round,1 ,vel_pred_test,7)
pos_pred_test = np.apply_along_axis(np.round,1 ,pos_pred_test,7)
z_GNSS = np.apply_along_axis(np.round,1,z_GNSS,7)

# %% 
leverarm = np.zeros(3)
S_a = np.eye(3)
# S_a = np.array(([1.06000000000000,	0.0150000000000000,	0.00600000000000000],
# [-0.00100000000000000,	0.999000000000000,	-0.0160000000000000],
# [0.0120000000000000,	-0.00500000000000000,	1.00100000000000]))


S_g = np.eye(3)
# S_g = np.array(([1.01000000000000,	0.0120000000000000,	0.00300000000000000],
# [-0.00200000000000000,	0.998000000000000,	-0.0210000000000000],
# [0.0180000000000000	,-0.00800000000000000	,1.00500000000000]))



# %% Define Beacon positions
"""
Lets start with 3 beacons spread out on 3 positions consisting of a matrix

Beacon_pos = [(beacon_1_x, beacon_1_y, beacon_1_z),
              (beacon_2_x, beacon_2_y, beacon_2_z),
              (beacon_3_x, beacon_3_y, beacon_3_z)]
"""
#Organized such that the first one has lowest y-value
beacon_location = np.array((
                           [30,-10,-0.5],
                           [-10,0,0.3],
                           [20,0,-0.3],
                           [15,10,1],
                           [20,15,1],
                           [-30,20,5],
                           [-15,25,3],
                           [15,35,-0.8],
                           [-5,40,0.5],
                           [25,40,0.8],
                           [10,50,-0.5],
                           [5,55,0.5],
                           [0,60,1],
                           [20,70,0.5],
                           [-20,70,1],))

print ("x_true[0]: \n", x_true[0])
# with open("../data/simulation_params_comb_maneuver.mat","wb") as mat_file:
#     scipy.io.savemat(mat_file, {'leverarm': leverarm.T})
#     scipy.io.savemat(mat_file, {'S_a': S_a.T})
#     scipy.io.savemat(mat_file, {'S_g': S_g.T})
#     scipy.io.savemat(mat_file, {'beacon_location': beacon_location.T})
    
#     scipy.io.savemat(mat_file, {'x_true': x_true.T})
#     scipy.io.savemat(mat_file, {"acc_t": acc_t.T})
#     scipy.io.savemat(mat_file,  {"timeIMU": timeIMU})
#     scipy.io.savemat(mat_file, {"timeGNSS": timeGNSS})
#     scipy.io.savemat(mat_file, {"z_acc": z_acc_vector.T})
#     scipy.io.savemat(mat_file, {"z_gyro": z_gyro_vector.T})
#     scipy.io.savemat(mat_file, {"z_GNSS": z_GNSS.T})
    
    
    # scipy.io.savemat(mat_file, {"omega_t": rollVelTrue.T})
    # scipy.io.savemat(mat_file, {"eul_att_t": eul_att.T})
    # scipy.io.savemat(mat_file, {"vel_pred_test": vel_pred_test.T})
    # scipy.io.savemat(mat_file, {"pos_pred_test": pos_pred_test.T})
    # %%
with open("../data/simulation_params_comb_maneuver_long_ver2.mat","wb") as mat_file:
    scipy.io.savemat(mat_file, {'leverarm': leverarm.T})
    scipy.io.savemat(mat_file, {'S_a': S_a.T})
    scipy.io.savemat(mat_file, {'S_g': S_g.T})
    scipy.io.savemat(mat_file, {'beacon_location': beacon_location})
    
    scipy.io.savemat(mat_file, {'x_true': x_true.T})
    scipy.io.savemat(mat_file, {"acc_t": acc_t.T})
    scipy.io.savemat(mat_file,  {"timeIMU": timeIMU})
    scipy.io.savemat(mat_file, {"timeGNSS": timeGNSS})
    scipy.io.savemat(mat_file, {"z_acc": z_acc_vector.T})
    scipy.io.savemat(mat_file, {"z_gyro": z_gyro_vector.T})
    scipy.io.savemat(mat_file, {"z_GNSS": z_GNSS.T})
    scipy.io.savemat(mat_file, {"omega_t": rollVelTrue.T})
    
