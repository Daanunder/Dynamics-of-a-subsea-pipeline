# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:45:55 2022

@author: thijs
"""
# start of FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scp
import scipy.integrate
# Input parameters
Mass_Ship = 500000            # [kg]

# Cable
M_p = 100000                  # [kg/m]
EI_pipe = 1e7                 # [N.m2]
EA_pipe = 1e6                 # [N]
Top_cable = 0                 # [m]
Base_cable = 3000             # [m]

# Define load parameters
f0 = 2                          # [Hz]
A0 = 0.1                        # [m]
T0 = 20                         # [s]

# Define output time vector
dt = 0.01                       # [s]
T = np.arange(0, 5*T0, dt)
nT = len(T)

# Define node coordinates
N_nodes = 3000
NodeCoord = np.zeros((N_nodes, 2))
NodeCoord[:, 0] = np.linspace(Top_cable, Base_cable, N_nodes)
print(NodeCoord)


# Define elements (and their properties


#             NodeLeft    NodeRight          m         EA        EI
N_elements = N_nodes - 1
m_pipe = M_p/N_elements
Elements = np.zeros((N_elements, 5), dtype=np.int64)
for idx in range(N_elements):
    Elements[idx, 0] = idx+1 
    Elements[idx, 1] = idx+2
    Elements[idx, 2] = m_pipe
    Elements[idx, 3] = EA_pipe
    Elements[idx, 4] = EI_pipe
print(Elements)

plt.figure()
for iElem in np.arange(0, N_elements):
    NodeLeft = Elements[iElem][0]-1
    NodeRight = Elements[iElem][1]-1

    c='black'
    plt.plot([NodeCoord[NodeLeft][1], NodeCoord[NodeRight][1]], [NodeCoord[NodeLeft][0], NodeCoord[NodeRight][0]], c=c)
    plt.plot([NodeCoord[NodeLeft][1], NodeCoord[NodeRight][1]], [NodeCoord[NodeLeft][0], NodeCoord[NodeRight][0]], 'ro', markersize=2)
                                                                 
plt.axis('equal')
plt.gca().invert_yaxis()
plt.xlim(-100, 100)
plt.grid()

