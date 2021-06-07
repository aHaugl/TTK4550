# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:02:30 2021

@author: Andreas
"""


import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__)) #test
parentdir = os.path.dirname(currentdir)  #src
sys.path.append(parentdir)

import quaternion
import numpy as np
import numpy.testing
import unittest



class Testquat(unittest.TestCase):
    
    def test_zero_angle(self):
        angle1 = np.array([0,0,0])
        quat1 = np.array([1,0,0,0])
        quat2 = np.array([1,0,1,0])
        
        trueangle1 = quaternion.quaternion_to_euler(quat1)
        truequat1 = quaternion.euler_to_quaternion(trueangle1)

        # self.assertAlmost_Equal(angle1, trueangle1)
        # self.assertAlmostEqual(quat1, truequat1)
        
        np.testing.assert_almost_equal(angle1, trueangle1)
        np.testing.assert_almost_equal(quat1,truequat1)
        # np.testing.assert_almost_equal(quat1,quat2)

    def test_gimbal_lock(self):
        gimbal_angle = np.array([np.pi/2,0,0]) #phi, theta, psi
        
        quat = quaternion.euler_to_quaternion(gimbal_angle)
        # print(quat)
        quat_back_to_euler = quaternion.quaternion_to_euler(quat) #output in rad?
                
        np.testing.assert_almost_equal(gimbal_angle, 
                                       quat_back_to_euler)



    def test_quat_prod(self):
        q_ans = np.array([0.682962, 0.682962, -0.183113, 0.183113])
        ql = np.array([0.707,0.707, 0, 0])
        qr = np.array([0.966, 0, 0, 0.259])
        quat_prod = quaternion.quaternion_product(ql,qr)
        # print(quat_prod)

        np.testing.assert_almost_equal(quat_prod,q_ans)
        
if __name__ == '__main__': unittest.main()