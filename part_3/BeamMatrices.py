import numpy as np
import math
def BeamMatricesJacket(m, EA, EI, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl], [xr, yr])      
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2)    # length
    alpha = math.atan2(yr - yl, xr - xl)        # angle

    # 2 - calculate transformation matrix T
    C = np.cos(alpha)
    S = np.sin(alpha)
    T = np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))
    

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, -EA/L, 0, 0], 
                    [0, 12*EI/L3, 6*EI/L2, 0, -12*EI/L3, 6*EI/L2], 
                    [0, 6*EI/L2, 4*EI/L, 0, -6*EI/L2, 2*EI/L], 
                    [-EA/L, 0, 0, EA/L, 0, 0], 
                    [0, -12*EI/L3, -6*EI/L2, 0, 12*EI/L3, -6*EI/L2], 
                    [0, 6*EI/L2, 2*EI/L, 0, -6*EI/L2, 4*EI/L]])
    M = m*L/420*np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])

    # 4 - rotate the matrices
    K = np.matmul(T, np.matmul(K, np.transpose(T)))
    M = np.matmul(T, np.matmul(M, np.transpose(T)))
    return M, K