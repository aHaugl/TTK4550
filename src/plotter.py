# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:25:10 2021

@author: andhaugl
"""
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
import scipy.stats

from scipy.stats.distributions import chi2


from utils import wrap_to_pi_from_euler, wrap_to_pi
from cat_slice import CatSlice
from quaternion import quaternion_to_euler

POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)
ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)



# %% Can be used to plot error covariance
def plot_error_pos_sigma(x_est, x_true, P_est, N):

    fig13, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    pos_err = x_true[:N, :3] - x_est[:N, :3]
    #pose_err[:, 2] *= 180/np.pi
    ylabels = ['m', 'm', 'm']
    tags = ['North error', 'East error', 'Down error']
    std = 3*np.sqrt(P_est[:N,POS_IDX,POS_IDX])
    # three_std = 3*np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_est[:N,:3]]))
    # std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, pos_err.T, std.T, tags, ylabels):
        ax.plot(err, label='error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated 3xSTD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    ax.set_xlabel("Steps [Seconds/0.01]")
    fig13.tight_layout()

def plot_error_vel_sigma(x_est, x_true, P_est, N):
    fig14, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    pos_err = x_true[:N, 3:6] - x_est[:N, 3:6]
    #pose_err[:, 2] *= 180/np.pi
    ylabels = ['m/s', 'm/s', 'm/s']
    tags = ['North vel error', 'East vel error', 'Down vel error']
    std = 3*np.sqrt(P_est[:N,VEL_IDX,VEL_IDX])
    # three_std = 3*np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_est[:N,3:6]]))
    # std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, pos_err.T, std.T, tags, ylabels):
        ax.plot(err, label='error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated 3xSTD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    ax.set_xlabel("Steps [Seconds/0.01]")
    fig14.tight_layout()
    
def plot_error_att_sigma(x_est, x_true, P_est, N):
    
    fig15, ax = plt.subplots(nrows=3, ncols=1, sharex=True
                             )
    x_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])
    x_est = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
    
    att_err = x_true[:N] - x_est[:N]
    #pose_err[:, 2] *= 180/np.pi
    ylabels = ['deg', 'deg', 'deg']
    tags = ['Pitch error', 'Roll error', 'Yaw error']
    
    std = 3*np.sqrt(P_est[:N,ERR_ATT_IDX,ERR_ATT_IDX])
    
    
    # three_std = 3*np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_est[:N,6:9]]))
    # std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, att_err.T, std.T, tags, ylabels):
        ax.plot(err, label='error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated 3xSTD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    ax.set_xlabel("Steps [Seconds/0.01]")
    fig15.tight_layout()
    
def plot_error_acc_bias_sigma(x_est, x_true, P_est, N):
    
    fig16, ax = plt.subplots(nrows=3, ncols=1, sharex=True
                             )
    x_true = x_true[:N, ERR_ACC_BIAS_IDX]
    x_est = x_est[:N, ERR_ACC_BIAS_IDX]
    
    acc_bias_err = x_true[:N] - x_est[:N]
    #pose_err[:, 2] *= 180/np.pi
    ylabels = ['m/s^2', 'm/s^2', 'm/s^2']
    tags = ['North acc_bias error', 'East acc_bias error', 'Down acc_bias error']
    
    std = 3*np.sqrt(P_est[:N,ERR_ACC_BIAS_IDX,ERR_ACC_BIAS_IDX])
    
    
    # three_std = 3*np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_est[:N,6:9]]))
    # std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, acc_bias_err.T, std.T, tags, ylabels):
        ax.plot(err, label='error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated 3xSTD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    ax.set_xlabel("Steps [Seconds/0.01]")
    fig16.tight_layout()
    
def plot_error_rate_bias_sigma(x_est, x_true, P_est, N):
    
    fig17, ax = plt.subplots(nrows=3, ncols=1, sharex=True
                             )
    x_true = x_true[:N, ERR_GYRO_BIAS_IDX]
    x_est =  x_est[:N, ERR_GYRO_BIAS_IDX]
    
    gyro_bias_err = x_true[:N] - x_est[:N]
    #pose_err[:, 2] *= 180/np.pi
    ylabels = ['deg', 'deg', 'deg']
    tags = ['Pitch gyro_bias error', 'Roll gyro_bias error', 'Yaw gyro_bias error']
    
    std = 3*np.sqrt(P_est[:N,ERR_GYRO_BIAS_IDX,ERR_GYRO_BIAS_IDX])
    
    
    # three_std = 3*np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_est[:N,6:9]]))
    # std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, gyro_bias_err.T, std.T, tags, ylabels):
        ax.plot(err, label='Gyro_bias error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated 3xSTD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    ax.set_xlabel("Steps [Seconds/0.01]")
    fig17.tight_layout()
    
# %% 
    

def plot_pos(t,N, x_est, tGNSS, GNSSk, z_GNSS, x_true=None):
    x_est = x_est[:, POS_IDX]
    x_true = x_true[:, POS_IDX]
    

    fig1, axs1 = plt.subplots(4, 1, num=1, clear=True)
    
    axs1[0].plot(t, x_est[:N,0])
    if x_true is not None:
        axs1[0].plot(t, x_true[:N,0],linestyle='dashed')
        
    axs1[0].scatter(tGNSS[:GNSSk],z_GNSS[:GNSSk, 0])
    axs1[0].set(ylabel="N position [m]")
    axs1[0].legend(["Estimated N pos", "True N pos"])
    
    axs1[1].plot(t, x_est[:N,1])
    if x_true is not None:
        axs1[1].plot(t, x_true[:N,1],linestyle='dashed')
    axs1[1].scatter(tGNSS[:GNSSk],z_GNSS[:GNSSk, 1])
    axs1[1].set(ylabel="E position [m]")
    axs1[1].legend(["Estimated N pos", "True N pos"])
    
    axs1[2].plot(t, x_est[:N,2])
    if x_true is not None:
        axs1[2].plot(t, x_true[:N,2],linestyle='dashed')
    axs1[2].scatter(tGNSS[:GNSSk],z_GNSS[:GNSSk, 2])
    axs1[2].set(ylabel="D position [m]")
    axs1[2].legend(["Estimated altitude", "True altitude"])
    
    if x_true is not None:
        axs1[3].plot(t, (x_true[:N] - x_est[:N] ))
        axs1[3].set(ylabel="Position error [m]")
        axs1[3].legend(
            [
                f"North ({np.sqrt(np.mean((x_true[:N,0]-x_est[:N,0]) ** 2)):.2e})",
                f"East ({np.sqrt(np.mean((x_true[:N,1]-x_est[:N,1]) ** 2)):.2e})",
                f"Down ({np.sqrt(np.mean((x_true[:N,2]-x_est[:N,2]) ** 2)):.2e})",
            ]
        )
    axs1[3].set(xlabel="Time [Seconds]")
    fig1.suptitle("Estimated vs true position")
    fig1.tight_layout()

  
    

def plot_vel(t,N, x_est, x_true=None):
    x_est = x_est[:, VEL_IDX]
    x_true = x_true[:, VEL_IDX]
    
    fig2, axs2 = plt.subplots(4, 1, num=2, clear=True)

    axs2[0].plot(t, x_est[:N,0])
    if x_true is not None:
        axs2[0].plot(t, x_true[:N,0],linestyle='dashed')
    axs2[0].set(ylabel="N velocity [m/s]")
    axs2[0].legend(["Est vel", "Truevel"])
    
    axs2[1].plot(t, x_est[:N,1])
    if x_true is not None:
        axs2[1].plot(t, x_true[:N,1],linestyle='dashed')
    axs2[1].set(ylabel="E velocity [m/s]")
    axs2[1].legend(["Est vel", "True vel"])
    

    axs2[2].plot(t, x_est[:N,2])
    if x_true is not None:
        axs2[2].plot(t, x_true[:N,2],linestyle='dashed')
    axs2[2].set(ylabel="D velocity [m/s]")
    axs2[2].legend(["Est vel", "True vel"])
    
    if x_true is not None:
        axs2[3].plot(t, (x_true[:N] - x_est[:N] ))
        axs2[3].set(ylabel="velocity error [m/s]")
        axs2[3].legend(
            [
                f"North ({np.sqrt(np.mean((x_true[:N,0] - x_est[:N,0]) ** 2)):.2e})",
                f"East ({np.sqrt(np.mean((x_true[:N,1] - x_est[:N,1]) ** 2)):.2e})",
                f"Down ({np.sqrt(np.mean((x_true[:N,2] - x_est[:N,2]) ** 2)):.2e})",
            ]
        )
    axs2[3].set(xlabel="Time [Seconds]")
    fig2.suptitle("Estimated vs true velocity")
    fig2.tight_layout()

def plot_acc(t,N, acc_meas, acc_t):
    fig3, axs3 = plt.subplots(4, 1, num=3, clear=True)
   
    axs3[0].plot(t, acc_meas[:N,0])
    axs3[0].plot(t, acc_t[:N,0],linestyle='dashed')
    axs3[0].set(ylabel="Surge acceleration [m/s^2]")
    axs3[0].legend(["Est acc", "True acc"])
    
    axs3[1].plot(t, acc_meas[:N,1])
    axs3[1].plot(t, acc_t[:N,1],linestyle='dashed')
    axs3[1].set(ylabel="Sway acceleration [m/s^2]")
    axs3[1].legend(["Est acc", "True acc"])
    
    axs3[2].plot(t, acc_meas[:N,2])
    axs3[2].plot(t, acc_t[:N,2],linestyle='dashed')
    axs3[2].set(ylabel="Heave acceleration [m/s^2]")
    axs3[2].legend(["Est acc", "True acc"])
    
    axs3[3].plot(t, (acc_t[:N] - acc_meas[:N] ))
    axs3[3].set(ylabel="acceleration error [m/s^2]")
    axs3[3].legend(
        [
            f"North ({np.sqrt(np.mean((acc_t[:N,0]-acc_meas[:N,0]) ** 2)):.2e})",
            f"East ({np.sqrt(np.mean((acc_t[:N,1]-acc_meas[:N,1]) ** 2)):.2e})",
            f"Down ({np.sqrt(np.mean((acc_t[:N,2]-acc_meas[:N,2]) ** 2)):.2e})",
        ]
    )
    axs3[3].set(xlabel="Time [Seconds]")
    fig3.suptitle("Measured (w/bias & noise) vs true acceleration")
    fig3.tight_layout()
    
def plot_gyro(t,N, measured_gyro, omega_t=None):
    
    # omega_t_rad = np.deg2rad(omega_t)
    # measured_gyro_rad = np.deg2rad(measured_gyro)

    fig4, axs4 = plt.subplots(4, 1, num=4, clear=True)
    
    
    axs4[0].plot(t, measured_gyro[:N,0])
    if omega_t is not None: 
        axs4[0].plot(t, omega_t[:N,0],linestyle='dashed')
    
    axs4[0].set(ylabel="Roll rate [rad/s]")
    axs4[0].legend(["Measured pitch rate", "True pitch rate"])
    
    axs4[1].plot(t, measured_gyro[:N,1])
    if omega_t is not None: 
        axs4[1].plot(t, omega_t[:N,1],linestyle='dashed')
    axs4[1].set(ylabel="Pitch rate [rad/s]")
    axs4[1].legend(["Measured roll rate", "True roll rate"])
    
    axs4[2].plot(t, measured_gyro[:N,2])
    if omega_t is not None: 
        axs4[2].plot(t, omega_t[:N,2],linestyle='dashed')
    axs4[2].set(ylabel="Yaw rate [rad/s]")
    axs4[2].legend(["Measured yaw rate", "True yaw rate"])
    
    if omega_t is not None:
        axs4[3].plot(t, (omega_t[:N] - measured_gyro[:N] ))
        axs4[3].set(ylabel="Angle rate error [rad/s]")
        axs4[3].legend(
            [
                f"North ({np.sqrt(np.mean((omega_t[:N,0] - measured_gyro[:N,0]) ** 2)):.2e})",
                f"East ({np.sqrt(np.mean((omega_t[:N,1] - measured_gyro[:N,1]) ** 2)):.2e})",
                f"Down ({np.sqrt(np.mean((omega_t[:N,2] - measured_gyro[:N,2]) ** 2)):.2e})",
            ]
        )
    axs4[3].set(xlabel="Time [Seconds]")
    fig4.suptitle("Estimated vs true angular rate")
    fig4.tight_layout()


def plot_angle(t,N, x_est, x_true=None):
    """
    Ground true is generated by equation 2.39 in Fossen, and not from quaternions.
    """
    # measured_angle = wrap_to_pi(measured_angle)
    # pi_wrap_eul = np.apply_along_axis(wrap_to_pi, 1, euler_est)
    
    x_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])
    x_est = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
    # eul = np.apply_along_axis(quaternion_to_euler, 1, x_est)
    
    fig5, axs5 = plt.subplots(4, 1, num=5, clear=True)
    
    axs5[0].plot(t, x_est[:N,0])
    if x_true is not None: 
        axs5[0].plot(t, x_true[:N,0],linestyle='dashed')
    axs5[0].set(ylabel=" Pitch [rad]")
    axs5[0].legend(["Est pitch", "True pitch"])
    
    axs5[1].plot(t, x_est[:N,1])
    if x_true is not None: 
        axs5[1].plot(t, x_true[:N,1],linestyle='dashed')
    axs5[1].set(ylabel="Roll [rad]")
    axs5[1].legend(["Est roll", "True roll"])
    
    axs5[2].plot(t, x_est[:N,2])
    if x_true is not None: 
        axs5[2].plot(t, x_true[:N,2],linestyle='dashed')
    axs5[2].set(ylabel="Yaw [rad]")
    axs5[2].legend(["Est yaw", "True raw"])
    
    if x_true is not None:
        axs5[3].plot(t, (x_true[:N] - x_est[:N] ))
        axs5[3].set(ylabel="Angular error [rad]")
        axs5[3].legend(
            [
                f"Pitch ({np.sqrt(np.mean((x_true[:N,0] - x_est[:N,0]) ** 2)):.2e})",
                f"Roll ({np.sqrt(np.mean((x_true[:N,1] - x_est[:N,1]) ** 2)):.2e})",
                f"Yaw ({np.sqrt(np.mean((x_true[:N,2] - x_est[:N,2]) ** 2)):.2e})",
            ]
        )
    axs5[3].set(xlabel="Time [Seconds]")
    fig5.suptitle("Estimated attitude vs true attitude")
    fig5.tight_layout()

    
    
def plot_estimate(t, N, x_est):
    fig6, axs6 = plt.subplots(5, 1, num=6, clear=True)

    eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])

    axs6[0].plot(t, x_est[:N, POS_IDX])
    axs6[0].set(ylabel="NED position [m]")
    axs6[0].legend(["North", "East", "Down"])

    axs6[1].plot(t, x_est[:N, VEL_IDX])
    axs6[1].set(ylabel="Velocities [m/s]")
    axs6[1].legend(["North", "East", "Down"])

    axs6[2].plot(t, eul[:N] * 180 / np.pi)
    axs6[2].set(ylabel="Euler angles [deg]")
    axs6[2].legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    axs6[3].plot(t, x_est[:N, ACC_BIAS_IDX])
    axs6[3].set(ylabel="Accl bias [m/s^2]")
    axs6[3].legend(["$x$", "$y$", "$z$"])

    axs6[4].plot(t, x_est[:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
    axs6[4].set(ylabel="Gyro bias [deg/h]")
    axs6[4].legend(["$x$", "$y$", "$z$"])
   
    axs6[3].set(xlabel="Time [Seconds]")
    fig6.suptitle("States estimates")
    fig6.tight_layout()
    # if dosavefigures:
    #     fig2.savefig(figdir+"state_estimates.pdf")


def plot_3Dpath(t, N, beacon_location, GNSSk, z_GNSS,  x_est, x_true=None):
    # 3d position plot
    x_est = x_est[:, POS_IDX]
    if x_true is not None:
        x_true = x_true[:, POS_IDX]
    
    fig7 = plt.figure(7)
    ax = fig7.add_subplot(1, 1, 1, projection='3d')

    
    ax.plot3D(x_est[:N,1], x_est[:N,0], x_est[:N,2])
    if x_true is not None:
        ax.plot3D(x_true[:N,1], x_true[:N,0], x_true[:N,2],linestyle='dashed')

    ax.scatter3D(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], z_GNSS[:GNSSk, 2])
    ax.scatter3D(beacon_location[:,1], beacon_location[:,0],beacon_location[:,2], marker = 'D')
    
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")
    ax.legend(["Estimated NED pos", "True NED pos", "z_GNSS", "Beacon location"])
    
    fig7.suptitle("Estimated vs true 3d path")
    fig7.tight_layout()
    # if dosavefigures:
    #     fig1.savefig(figdir+"ned.pdf")

    # state estimation
    
def plot_path(t,N, beacon_location, GNSSk, z_GNSS, x_est, x_true=None):
    #
    x_est = x_est[:, POS_IDX]
    x_true = x_true[:, POS_IDX]
    fig8 = plt.figure(8)
    ax8 = fig8.add_subplot(1,1,1)

    ax8.plot(x_est[:N,1], x_est[:N,0])
    if x_true is not None:
        ax8.plot(x_true[:N,1], x_true[:N,0],linestyle='dashed')    
    ax8.scatter(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], marker=',')
    ax8.scatter(beacon_location[:,1], beacon_location[:,0], marker = 'D')
        
    ax8.set(ylabel="NED position [m]")
    ax8.set_xlabel("East [m]")
    ax8.set_ylabel("North [m]")
    ax8.legend(["Estimated NED pos", "True NED pos", "z_GNSS", "Beacon location"])
    
    fig8.suptitle("Estimated vs true path")
    fig8.tight_layout()
    
def state_error_plots(t, N, x_est, x_true, delta_x):
    if x_true is None:
        print('coud not plot error as xtrue is None')
        return
    fig9, axs9 = plt.subplots(5, 1, num=9, clear=True)
    eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
    eul_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])

    # TODO use this in legends
    delta_x_RMSE = np.sqrt(np.mean(delta_x[:N] ** 2, axis=0))
    axs9[0].plot(t, delta_x[:N, POS_IDX])
    axs9[0].set(ylabel="NED position error [m]")
    axs9[0].legend(
        [
            f"North ({np.sqrt(np.mean(delta_x[:N, POS_IDX[0]]**2)):.2e})",
            f"East ({np.sqrt(np.mean(delta_x[:N, POS_IDX[1]]**2)):.2e})",
            f"Down ({np.sqrt(np.mean(delta_x[:N, POS_IDX[2]]**2)):.2e})",
        ]
    )

    axs9[1].plot(t, delta_x[:N, VEL_IDX])
    axs9[1].set(ylabel="Velocities error [m]")
    axs9[1].legend(
        [
            f"North ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[0]]**2)):.2e})",
            f"East ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[1]]**2)):.2e})",
            f"Down ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[2]]**2)):.2e})",
        ]
    )

    eul_error = wrap_to_pi(eul_true[:N]- eul[:N]) * 180 / np.pi
    axs9[2].plot(t, eul_error)
    axs9[2].set(ylabel="Euler angles error [deg]")
    axs9[2].legend(
        [
            rf"$\phi$ ({np.sqrt(np.mean((eul_error[:N, 0])**2)):.2e})",
            rf"$\theta$ ({np.sqrt(np.mean((eul_error[:N, 1])**2)):.2e})",
            rf"$\psi$ ({np.sqrt(np.mean((eul_error[:N, 2])**2)):.2e})",
        ]
    )

    axs9[3].plot(t, delta_x[:N, ERR_ACC_BIAS_IDX])
    axs9[3].set(ylabel="Accl bias error [m/s^2]")
    axs9[3].legend(
        [
            f"$x$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[0]]**2)):.2e})",
            f"$y$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[1]]**2)):.2e})",
            f"$z$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[2]]**2)):.2e})",
        ]
    )

    axs9[4].plot(t, delta_x[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi)
    axs9[4].set(ylabel="Gyro bias error [deg/s]")
    axs9[4].legend(
        [
            f"$x$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[0]]* 180 / np.pi)**2)):.2e})",
            f"$y$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[1]]* 180 / np.pi)**2)):.2e})",
            f"$z$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[2]]* 180 / np.pi)**2)):.2e})",
        ]
    )
    axs9[4].set(xlabel="Time [Seconds]")
    fig9.suptitle("States estimate errors")
    fig9.tight_layout()
    # if dosavefigures:
    #     fig9.savefig(figdir+"state_estimate_errors.pdf")


def error_distance_plot(t, N, dt, GNSSk, x_true, delta_x, z_GNSS):
    # 3d position plot
    fig10, axs10 = plt.subplots(2, 1, num=10, clear=True)

    pos_err = np.linalg.norm(delta_x[:N, POS_IDX], axis=1)
    meas_err = np.linalg.norm(
        x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk], axis=1)
    axs10[0].plot(t, pos_err)
    axs10[0].plot(np.arange(0, N, 100) * dt, meas_err)

    axs10[0].set(ylabel="Position error [m]")
    axs10[0].legend(
        [
            f"Estimation error ({np.sqrt(np.mean(np.sum(delta_x[:N, POS_IDX]**2, axis=1)))})",
            f"Measurement error ({np.sqrt(np.mean(np.sum((x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk])**2, axis=1)))})",
        ]
    )

    axs10[1].plot(t, np.linalg.norm(delta_x[:N, VEL_IDX], axis=1))
    axs10[1].set(ylabel="Speed error [m/s]")
    axs10[1].legend(
        [f"RMSE: {np.sqrt(np.mean(np.sum(delta_x[:N, VEL_IDX]**2, axis=0)))}"])
    
    # axs10.set(xlabel="Time [Seconds]")
    fig10.tight_layout()
    # if dosavefigures:
    #     fig10.savefig(figdir+"error_distance_plot.pdf")

    # %% Consistency


def plot_NEES(t, N, dt,
              NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias,
              NEES_gyrobias, confprob=0.95):
    fig11, axs11 = plt.subplots(6, 1, num=11, clear=True)
  #  for ax in axs11:
       # ax.set_yscale('log')

    CI15 = np.array(scipy.stats.chi2.interval(confprob, 15 )).reshape((2, 1))
    CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))

    axs11[0].plot(t, (NEES_all[:N]).T)
    axs11[0].plot(np.array([0, N - 1]) * dt, (CI15 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI15[0] <= NEES_all[:N]) * (NEES_all[:N] <= CI15[1]))
    axs11[0].set(
        title=f"Total NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs11[1].plot(t, (NEES_pos[0:N]).T)
    axs11[1].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_pos[:N]) * (NEES_pos[:N] <= CI3[1]))
    axs11[1].set(
        title=f"Position NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs11[2].plot(t, (NEES_vel[0:N]).T)
    axs11[2].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_vel[:N]) * (NEES_vel[:N] <= CI3[1]))
    axs11[2].set(
        title=f"Velocity NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs11[3].plot(t, (NEES_att[0:N]).T)
    axs11[3].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_att[:N]) * (NEES_att[:N] <= CI3[1]))
    axs11[3].set(
        title=f"Attitude NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs11[4].plot(t, (NEES_accbias[0:N]).T)
    axs11[4].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_accbias[:N])
                       * (NEES_accbias[:N] <= CI3[1]))
    axs11[4].set(
        title=f"Accelerometer bias NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs11[5].plot(t, (NEES_gyrobias[0:N]).T)
    axs11[5].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_gyrobias[:N])
                       * (NEES_gyrobias[:N] <= CI3[1]))
    axs11[5].set(
        title=f"Gyro bias NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )
    
    axs11[5].set(xlabel="Time [Seconds]")
    fig11.tight_layout()
    # if dosavefigures:
    #     fig12.savefig(figdir+"nees_nis.pdf")

    # boxplot


def plot_NIS(
        NIS,
        confprob=0.95):

    fig12, ax12 = plt.subplots(1, sharex=True, num=12, clear=True)
    #ax12.set_yscale('log')
    Ts_list = NIS[:, 0]
    NIS_data = NIS[:, 1]

    CI3 = np.array(scipy.stats.chi2.interval(confprob, 3 )) 

    ax12.plot(Ts_list, NIS_data)
    ax12.plot([0, Ts_list[-1]], np.repeat(CI3[None], 2, 0), "--r")
    ax12.set_ylabel("NIS CV")
    inCIpos = np.mean((CI3[0] <= NIS_data) * (NIS_data <= CI3[1]))
    ax12.set_title(
        f"NIS CV, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")
    ax12.set(xlabel="Time [Seconds]")

def box_plot(N, GNSSk,
             NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias,
             NEES_gyrobias, NIS):
    fig13, axs13 = plt.subplots(1, 3, num = 13, clear =True)

    gauss_compare = np.sum(np.random.randn(3, GNSSk)**2, axis=0)
    axs13[0].boxplot([NIS[0:GNSSk], gauss_compare], notch=True)
    axs13[0].legend(['NIS', 'gauss'])
    plt.grid()

    gauss_compare_15 = np.sum(np.random.randn(15, N)**2, axis=0)
    axs13[1].boxplot([NEES_all[0:N].T, gauss_compare_15], notch=True)
    axs13[1].legend(['NEES', 'gauss (15 dim)'])
    plt.grid()

    gauss_compare_3 = np.sum(np.random.randn(3, N)**2, axis=0)
    axs13[2].boxplot([NEES_pos[0:N].T, NEES_vel[0:N].T, NEES_att[0:N].T,
                     NEES_accbias[0:N].T, NEES_gyrobias[0:N].T, gauss_compare_3], notch=True)
    axs13[2].legend(['NEES pos', 'NEES vel', 'NEES att',
                    'NEES accbias', 'NEES gyrobias', 'gauss (3 dim)'])
    plt.grid()

    axs13.tight_layout()
    # if dosavefigures:
    #     fig13.savefig(figdir+"boxplot.pdf")

