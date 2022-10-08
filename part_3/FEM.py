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

from BeamMatrices import BeamMatricesJacket
# Input parameters
A = 0.2
rho_cable = 7850*A

# Cable
M_p = 100000                  # [kg/m]
EI_pipe = 1e9                 # [N.m2]
EA_pipe = 0     # neglecting elongation
#EA_pipe = 210000000 * A       # [N]
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
N_nodes = 51
NodeCoord = np.zeros((N_nodes, 2))
NodeCoord[:, 0] = np.linspace(Top_cable, Base_cable, N_nodes)
print(NodeCoord)


# Define elements (and their properties


#             NodeLeft    NodeRight          m         EA        EI
N_elements = N_nodes - 1
m_pipe = rho_cable*(Base_cable-Top_cable)/N_elements
Elements = np.zeros((N_elements, 5), dtype=np.int64)
for idx in range(N_elements):
    Elements[idx, 0] = idx+1 
    Elements[idx, 1] = idx+2
    Elements[idx, 2] = m_pipe
    Elements[idx, 3] = EA_pipe
    Elements[idx, 4] = EI_pipe
print(Elements)

# visualize discretization of pipe
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

# assign DOF's to nodes
nDof = 3*N_nodes              # 3 Degrees of freedom per node
K = np.zeros((nDof*nDof))
M = np.zeros((nDof*nDof))

# Use preknown matrix for the local elemnts and set up the global matrix

for i, element in enumerate(Elements):
    # Get the nodes of the elements
    NodeLeft  = element[0]-1
    NodeRight = element[1]-1
    
    # Get the degrees of freedom that correspond to each node
    Dofs_Left  = 3*(NodeLeft)  + np.arange(0, 3)
    Dofs_Right = 3*(NodeRight) + np.arange(0, 3)

    # Get the properties of the element
    m  = element[2]
    EA = element[3]
    EI = element[4]

    # Calculate the local matrix of each element
    Me, Ke = BeamMatricesJacket(m, EA, EI, ([NodeCoord[NodeLeft][0], NodeCoord[NodeLeft][1]], [NodeCoord[NodeRight][0], NodeCoord[NodeRight][1]]))

    # Assemble the matrices at the correct place
    nodes = np.append(Dofs_Left, Dofs_Right)
    for i in np.arange(0, 6):
        for j in np.arange(0, 6):
            ij = nodes[j] + nodes[i]*nDof
            #print(ij)
            M[ij] = M[ij] + Me[i, j]
            K[ij] = K[ij] + Ke[i, j]
            
# Reshape the global matrix from a 1-dimensional array to a 2-dimensional array
M = M.reshape((nDof, nDof))
K = K.reshape((nDof, nDof))

# Look at the matrix structure, we expect purely diagonal matrix, since our structure is a 1D element
plt.figure()
plt.spy(M)
plt.title("Mass matrix")
plt.figure()
plt.spy(K)
plt.title("Stiffness matrix")

# ---------------------------- Boundary Conditions ---------------------------#
# describe which DOF's are free and which are prescribed through the boundary conditions          # prescribed DOFs
DofsP = np.array([0, 1, nDof-3, nDof-2])
DofsF = np.arange(0, nDof)       #  Initially set all DOF's as free DOf's
DofsF = np.delete(DofsF, DofsP)  # remove the fixed DOFs from the free DOFs array

# free & fixed array indices
fx = DofsF[:, np.newaxis]
fy = DofsF[np.newaxis, :]
bx = DofsP[:, np.newaxis]
by = DofsP[np.newaxis, :]

# Mass
M_FF = M[fx, fy]
M_FP = M[fx, by]
M_PF = M[bx, fy]
M_PP = M[bx, by]

# Stiffness
K_FF = K[fx, fy]
K_FP = K[fx, by]
K_PF = K[bx, fy]
K_PP = K[bx, by]


## Start Modal Analysis

# compute eigenvalues and eigenvectors
mat = np.linalg.inv(M_FF) @ K_FF
w2, vr = np.linalg.eig(mat)

w = np.sqrt(w2.real)
f = w/2/np.pi
print(len(f))
print(np.min(f[f!= 0]))
idx = f.argsort() 
f = f[idx]

vr_sorted = vr[:, idx]
ModalShape = np.zeros((nDof, len(f)))
# mask to get the right modal shapes
mask = np.ones(ModalShape.shape[0],dtype=bool)
mask[DofsP]=0

ModalShape[mask,: ] = vr_sorted

print(vr_sorted.shape)

# ----------------- Define in how many modes you are interested --------------#
N_mode = 5 
# ----------------------------------------------------------------------------#

R_M = np.nonzero(f)[0][0]

plt.figure()
nCol = int(np.round(np.sqrt(N_mode)))
nRow = int(np.ceil(N_mode/nCol))
for iMode in np.arange(R_M, N_mode +R_M):
    plt.subplot(nRow, nCol, iMode-R_M+1)
    Shape = ModalShape[:, iMode]

    # Scale the mode such that maximum deformation is 10
    MaxTranslationx = np.max(np.abs(Shape[0::3]))

    MaxTranslationy = np.max(np.abs(Shape[1::3]))
    if MaxTranslationx != 0:
         Shape[0::3] = Shape[0::3]/MaxTranslationx*300
    Shape[1::3] = Shape[1::3]/MaxTranslationy*300

    # Get the deformed shape
    DisplacedNode = ([i[0] for i in NodeCoord] + Shape[0::3], [i[1] for i in NodeCoord] + Shape[1::3])
    for element in Elements:
        NodeLeft = element[0]-1
        NodeRight = element[1]-1
        m = element[2]
        EA = element[3]
        EI = element[4]
        c = 'black'
        plt.plot([DisplacedNode[0][NodeLeft], DisplacedNode[0][NodeRight]], 
                    [DisplacedNode[1][NodeLeft], DisplacedNode[1][NodeRight]], c=c)
    plt.title("Mode "+str(iMode-R_M)+": f = "+str(np.round(f[iMode], 3))+" Hz")


# automatically fix subplot spacing
plt.tight_layout()
