# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:45:55 2022

@author: thijs
"""
# start of FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scp
import scipy.integrate as scpi
from matplotlib.animation import FuncAnimation
from BeamMatrices import BeamMatricesJacket
# Input parameters
A = 0.2
rho_cable = 7850*A

# Cable
                 # [kg/m]
EI_pipe = 1e14               # [N.m2]
EA_pipe = 0                   # neglecting elongation
#EA_pipe = 210000000 * A      # [N]
Top_cable = 0                 #  [m]
Base_cable = 5000             # [m]

# Define load parameters
f0 = 2                          # [Hz]
A0 = 0.1                        # [m]
T0 = 20                         # [s]

# Define output time vector
dt = 0.01                       # [s]
T = np.arange(0, 5*T0, dt)
nT = len(T)

# Define node coordinates
N_nodes = 41
NodeCoord = np.zeros((N_nodes, 2))
NodeCoord[:, 0] = np.linspace(Top_cable, Base_cable, N_nodes)
print(NodeCoord)


# Define elements (and their properties


#             NodeLeft    NodeRight          m         EA        EI
N_elements = N_nodes - 1
dx = (Base_cable-Top_cable)/N_elements
m_pipe = rho_cable*dx
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
DofsP = np.array([0,  1, nDof-3, nDof-2])
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

idx = f.argsort() 
f = f[idx]

vr_sorted = vr[:, idx]
ModalShape = np.zeros((nDof, len(f)))
# mask to get the right modal shapes
mask = np.ones(ModalShape.shape[0],dtype=bool)
mask[DofsP]=0

ModalShape[mask,: ] = vr_sorted



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
    plt.title("Mode "+str(iMode-R_M+1)+": f = "+str(np.round(f[iMode], 3))+" Hz")


# automatically fix subplot spacing
plt.tight_layout()

# Compute Modal quantities and decouple the system
nMode = 15
print(vr_sorted)
PHI = vr_sorted[:,0+R_M:nMode+R_M]
print(PHI)

Mm = np.zeros(nMode)
Km = np.zeros(nMode)
Cm = np.zeros(nMode)
ModalDampRatio = 0.05
# Compute your "nMode" entries of the modal mass, stiffness and damping
for i in range(nMode):
    Mm[i] = PHI[:, i].T  @ M_FF @ PHI[:, i]
    Km[i] = PHI[:, i].T  @ K_FF @ PHI[:, i]
    Cm[i] = 2*ModalDampRatio*np.sqrt(Mm[i]*Km[i])
    print(f"Computing Mode: {i+1} \nMm = {Mm[i]:.3f}, Km = {Km[i]:.3f}, Cm = {Cm[i]:.3f}")


# forcing
# def ub(t, T0):
#     if t <= T0:
#         return A0*np.sin(2*np.pi*f0*t)*np.array([1, 0, 0, 1, 0, 0])
#     else:
#         return np.array([0, 0, 0, 0, 0, 0])
# def dub_dt2(t, T0):
#     return -(2*np.pi*f0)**2*ub(t, T0)

# trial forcing
c_F = 20 #kN
F = np.zeros(len(DofsF))
f = 100* np.sin(np.arange(Top_cable+dx, Base_cable, dx)*np.pi/2/Base_cable)
F[2::3] = f
print(F)
# time dependant forcing
def Feq(t):
      
    Fequ  = np.dot(PHI.T, F)
    return Fequ


print("Forcing function determined")


# solving in time domain

# update function
def qdot(t,q):
    Um = q[0:nMode]                                 # first entries are the displacemnt
    Vm = q[nMode:2*nMode]                           # second half is the velocity
    Am = ( Feq(t) - (Km * Um + Cm * Vm) ) / Mm      # compute the accelerations
    return np.append(Vm,Am)                         # return the velocity and accelerations

# Define the Initial Conditions
q0 = np.zeros(2*nMode)

# ------------------------- Solve the ODE in time using build-in solver ------------------ #  
q = scpi.solve_ivp(fun=qdot,y0=q0,t_span=[0,1.5e5])
# ---------------------------------------------------------------------------------------- #
print('Solving done')

plt.figure()
plt.plot(np.linspace(0,1e3, len(q.y[0])),q.y[1])

lent = len(q.t)

# back to dof
# Show the result on the original coordinates
U_F = np.zeros((lent,len(PHI[:,0])))
for iMode in range(nMode):
    for it in range(lent):
        U_F[it,:] += PHI[:, iMode] * q.y[iMode][it]
print("Back to original DoF's done")


mask2 = np.nonzero(np.arange(0, len(U_F[0, :])-1, 1) % 3)

# xiu = U_F[1600, mask2]
# print(xiu)
# print(q.t[1600])
# xiu = xiu[0, 1::2]
# plt.figure()

# plt.plot(np.arange(len(xiu)), xiu)

U = np.zeros((len(q.t), len(NodeCoord[1:-1].flatten())))
x = np.arange(0, len(U[0]), 2)
y = np.arange(1, len(U[1]), 2)
Ux = np.zeros((len(q.t), len(U[0])//2))
Uy = np.zeros((len(q.t),len(U[0])//2)) 
for t in range(len(q.t)):
        U[t,:] = U_F[t,mask2]+NodeCoord[1:-1].flatten()
        #print(U[t, :])
        Ux[t] = U[t, 0::2]
        Uy[t] = U[t, 1::2]


# add BC:
BC1x = np.ones(len(q.t))*Top_cable
BC1y = np.zeros(len(q.t))
BC2x = np.ones(len(q.t))*Base_cable
BC2y = np.zeros(len(q.t))
Ux = np.insert(Ux, 0, BC1x, axis=1)
Uy = np.insert(Uy, 0, BC1y, axis=1)
Ux = np.insert(Ux, N_nodes-1, BC2x, axis=1)
Uy = np.insert(Uy, N_nodes-1, BC2y, axis=1)
plt.figure()
plt.plot(Ux[20], Uy[20])

fig = plt.figure()

def animate(i):
    plt.cla()
    plt.title(f"Time is {q.t[i*10]:.1f} s")
    plt.plot(Ux[i], Uy[i])
    plt.ylim(-1, 1)

    anim = FuncAnimation(fig, animate, frames=len(q.t)//10,interval=1, repeat=False)


