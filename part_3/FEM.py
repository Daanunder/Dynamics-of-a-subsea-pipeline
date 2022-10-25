# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:45:55 2022

@author: thijs
"""
# start of FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.animation import FuncAnimation
from BeamMatrices import BeamMatricesJacket

class ContinousBeam(object):

    """Class that allows numerical simulation of a simple Euler-Bernoulli beam"""

    def __init__(self):
        """Sets up basic parameters and matrices"""
        

        # Input parameters
        self.A = 0.2
        self.rho_cable = 7850*self.A

        # Cable
                         # [kg/m]
        self.EI_pipe = 1e14                 # [N.m2]
        self.EA_pipe = 0                   # neglecting elongation
        #EA_pipe = 210000000 * A      # [N]
        self.top_cable = 0                 #  [m]
        self.base_cable = 5000             # [m]

        # Define load parameters
        self.f0 = 2                          # [Hz]
        self.A0 = 0.1                        # [m]
        self.T0 = 20                         # [s]

        # Define output time vector
        self.dt = 0.01                       # [s]
        self.T = np.arange(0, 5*self.T0, self.dt)
        self.nT = len(self.T)

        # Define node coordinates
        self.N_nodes = 41
        self.NodeCoord = np.zeros((self.N_nodes, 2))
        self.NodeCoord[:, 0] = np.linspace(self.top_cable, self.base_cable, self.N_nodes)

        # assign DOF's to nodes
        self.nDof = 3*self.N_nodes              # 3 Degrees of freedom per node

        # Modal analysis
        self.plot_n_modes = 5
        self.modal_damping_ratio = 0.05

        # Numerical parameters
        self.n_modes = 15
        self.q0 = np.zeros(2*self.n_modes)

        # ---------------------------- Boundary Conditions ---------------------------#
        # describe which DOF's are free and which are prescribed through the boundary conditions          # prescribed DOFs
        self.prescribed_dofs = np.array([0,  1, self.nDof-3, self.nDof-2])
        self.free_dofs = np.arange(0, self.nDof)       #  Initially set all DOF's as free DOf's
        self.free_dofs = np.delete(self.free_dofs, self.prescribed_dofs)  # remove the fixed DOFs from the free DOFs array
    
        self.discretize_beam()
        self.setup_system_matrices()
        self.apply_boundary_conditions()
        self.setup_modal_matrix()
    
    def append_equation_to_latex(self, eq, fname="eoms_latex.txt"):
        with open(fname, "a") as f:
            f.writelines([sp.latex(eq, itex=True, mode="equation", mul_symbol="dot"), "\n\n\n"])

    def discretize_beam(self):
        # Define elements (and their properties
        #             NodeLeft    NodeRight          m         EA        EI
        self.N_elements = self.N_nodes - 1
        self.dx = (self.base_cable-self.top_cable)/self.N_elements
        self.m_pipe = self.rho_cable*self.dx
        self.Elements = np.zeros((self.N_elements, 5), dtype=np.int64)
        for idx in range(self.N_elements):
            self.Elements[idx, 0] = idx
            self.Elements[idx, 1] = idx+1
            self.Elements[idx, 2] = self.m_pipe
            self.Elements[idx, 3] = self.EA_pipe
            self.Elements[idx, 4] = self.EI_pipe
        
    def plot_discretization(self):
        # visualize discretization of pipe
        plt.figure()
        for i in np.arange(0, self.N_elements):
            NodeLeft = self.Elements[i][0]
            NodeRight = self.Elements[i][1]

            c='black'
            plt.plot([self.NodeCoord[NodeLeft][1], self.NodeCoord[NodeRight][1]], [self.NodeCoord[NodeLeft][0], self.NodeCoord[NodeRight][0]], c=c)
            plt.plot([self.NodeCoord[NodeLeft][1], self.NodeCoord[NodeRight][1]], [self.NodeCoord[NodeLeft][0], self.NodeCoord[NodeRight][0]], 'ro', markersize=2)
                                                                         
        plt.axis('equal')
        plt.gca().invert_yaxis()
        plt.xlim(-100, 100)
        plt.grid()
        plt.show()

    
    def setup_system_matrices(self):
        self.K = np.zeros((self.nDof*self.nDof))
        self.M = np.zeros((self.nDof*self.nDof))

        # Use preknown matrix for the local elemnts and set up the global matrix

        for i, element in enumerate(self.Elements):
            # Get the nodes of the elements
            NodeLeft  = element[0]
            NodeRight = element[1]
            
            # Get the degrees of freedom that correspond to each node
            ## TODO: implement connectivity matrix function
            Dofs_Left  = 3*(NodeLeft)  + np.arange(0, 3)
            Dofs_Right = 3*(NodeRight) + np.arange(0, 3)

            # Get the properties of the element
            m  = element[2]
            EA = element[3]
            EI = element[4]

            # Calculate the local matrix of each element
            # TODO: Document on translation/rotation matrix
            Me, Ke = BeamMatricesJacket(m, EA, EI, ([self.NodeCoord[NodeLeft][0], self.NodeCoord[NodeLeft][1]], [self.NodeCoord[NodeRight][0], self.NodeCoord[NodeRight][1]]))

            # Assemble the matrices at the correct place
            nodes = np.append(Dofs_Left, Dofs_Right)
            for i in np.arange(0, 6):
                for j in np.arange(0, 6):
                    ij = nodes[j] + nodes[i]*self.nDof
                    #print(ij)
                    self.M[ij] = self.M[ij] + Me[i, j]
                    self.K[ij] = self.K[ij] + Ke[i, j]
                    
        # Reshape the global matrix from a 1-dimensional array to a 2-dimensional array
        self.M = self.M.reshape((self.nDof, self.nDof))
        self.K = self.K.reshape((self.nDof, self.nDof))
    
    def plot_system_matrices(self):
        # Look at the matrix structure, we expect purely diagonal matrix, since our structure is a 1D element
        plt.figure()
        plt.spy(self.M)
        plt.title("Mass matrix")
        plt.figure()
        plt.spy(self.K)
        plt.title("Stiffness matrix")
        plt.show()


    
    def apply_boundary_conditions(self):
        # free & fixed array indices [M x N]
        fn = self.free_dofs[:, np.newaxis]
        fm = self.free_dofs[np.newaxis, :]
        bn = self.prescribed_dofs[:, np.newaxis]
        bm = self.prescribed_dofs[np.newaxis, :]

        # Mass
        M_FF = self.M[fm, fn]
        M_FP = self.M[fm, bn]
        M_PF = self.M[bm , fn]
        M_PP = self.M[bm , bn]

        self.M_FF = M_FF

        # Stiffness
        K_FF = self.K[fm, fn]
        K_FP = self.K[fm, bn]
        K_PF = self.K[bm , fn]
        K_PP = self.K[bm , bn]

        self.K_FF = K_FF

        return M_FF, K_FF

    
    def setup_modal_matrix(self):
        # compute eigenvalues and eigenvectors
        mat = np.linalg.inv(self.M_FF) @ self.K_FF
        w2, vr = np.linalg.eig(mat)

        w = np.sqrt(w2.real)
        idx = w.argsort()
        w = w[idx]
        vr = vr[:, idx]
        f = w/2/np.pi

        self.eigen_frequencies = f
        self.eigen_values = w
        self.eigen_vectors = vr

        ModalShape = np.zeros((self.nDof, len(f)))

        # mask to get the right modal shapes
        mask = np.ones(ModalShape.shape[0],dtype=bool)
        mask[self.prescribed_dofs] = 0

        ModalShape[mask,: ] = vr
        self.first_mode = np.nonzero(f)[0][0]

        self.modal_shape = ModalShape
        return self.modal_shape

    def plot_modal_shapes(self):
        plt.figure()

        nCol = int(np.round(np.sqrt(self.plot_n_modes)))
        nRow = int(np.ceil(self.plot_n_modes/nCol))
        for iMode in np.arange(self.first_mode, self.plot_n_modes + self.first_mode):
            plt.subplot(nRow, nCol, iMode-self.first_mode+1)
            shape = self.modal_shape[:, iMode]

            # Scale the mode such that maximum deformation is 10
            MaxTranslationx = np.max(np.abs(shape[0::3]))

            MaxTranslationy = np.max(np.abs(shape[1::3]))
            if MaxTranslationx != 0:
                 shape[0::3] = shape[0::3]/MaxTranslationx*300
            shape[1::3] = shape[1::3]/MaxTranslationy*300

            # Get the deformed shape
            DisplacedNode = ([i[0] for i in self.NodeCoord] + shape[0::3], [i[1] for i in self.NodeCoord] + shape[1::3])
            for element in self.Elements:
                NodeLeft = element[0]-1
                NodeRight = element[1]-1
                m = element[2]
                EA = element[3]
                EI = element[4]
                c = 'black'
                plt.plot([DisplacedNode[0][NodeLeft], DisplacedNode[0][NodeRight]], 
                            [DisplacedNode[1][NodeLeft], DisplacedNode[1][NodeRight]], c=c)
            plt.title("Mode "+str(iMode-self.first_mode+1)+": f = "+str(np.round(self.eigen_frequencies[iMode], 3))+" Hz")

        # automatically fix subplot spacing
        plt.tight_layout()
        plt.show()

    def get_decoupled_system(self):
        # Compute Modal quantities and decouple the system
        contributing_modes = slice(self.first_mode, self.n_modes+self.first_mode)
        self.PHI = self.eigen_vectors[:,contributing_modes]

        self.Mm = np.zeros(self.n_modes)
        self.Km = np.zeros(self.n_modes)
        self.Cm = np.zeros(self.n_modes)

        # Compute your "self.n_modes" entries of the modal mass, stiffness and damping
        for i in range(self.n_modes):
            self.Mm[i] = self.PHI[:, i].T  @ self.M_FF @ self.PHI[:, i]
            self.Km[i] = self.PHI[:, i].T  @ self.K_FF @ self.PHI[:, i]
            self.Cm[i] = 2*self.modal_damping_ratio*np.sqrt(self.Mm[i]*self.Km[i])
            print(f"Computing Mode: {i+1} \nMm = {Mm[i]:.3f}, Km = {Km[i]:.3f}, Cm = {Cm[i]:.3f}")

    def get_forcing(self):
        c_F = 20 #kN
        F = np.zeros(len(self.free_dofs))
        f = 1000* np.sin(np.arange(self.top_cable+self.dx, self.base_cable, self.dx)*np.pi/self.base_cable)
        F[2::3] = f 
        Fequ  = np.dot(self.PHI.T, F)
        return Fequ
        # forcing
        # def ub(t, T0):
        #     if t <= T0:
        #         return A0*np.sin(2*np.pi*f0*t)*np.array([1, 0, 0, 1, 0, 0])
        #     else:
        #         return np.array([0, 0, 0, 0, 0, 0])
        # def dub_dt2(t, T0):
        #     return -(2*np.pi*f0)**2*ub(t, T0)

    # update function
    def qdot(t,q):
        Um = q[0:nMode]                                 # first entries are the displacemnt
        Vm = q[nMode:2*nMode]                           # second half is the velocity
        Am = ( self.get_forcing(t) - (self.Km * Um + self.Cm * Vm) ) / self.Mm      # compute the accelerations
        return np.append(Vm,Am)                         # return the velocity and accelerations

    # solving in time domain
    def get_solution_FEM(self):
        self.solution = scipy.integrate.solve_ivp(fun=self.qdot,y0=self.q0,t_span=[self.T[0],self.T[-1]])
        print('Solving done')
    
    


beam = ContinousBeam()

#plt.figure()
#plt.plot(np.linspace(0,1e3, len(q.y[0])),q.y[1])
#
#lent = len(q.t)
#
## back to dof
## Show the result on the original coordinates
#U_F = np.zeros((lent,len(PHI[:,0])))
#for iMode in range(nMode):
    #for it in range(lent):
        #U_F[it,:] += PHI[:, iMode] * q.y[iMode][it]
#print("Back to original DoF's done")
#
#
#mask2 = np.nonzero(np.arange(0, len(U_F[0, :])-1, 1) % 3)
#
## xiu = U_F[1600, mask2]
## print(xiu)
## print(q.t[1600])
## xiu = xiu[0, 1::2]
## plt.figure()
#
## plt.plot(np.arange(len(xiu)), xiu)
#
#U = np.zeros((len(q.t), len(NodeCoord[1:-1].flatten())))
#x = np.arange(0, len(U[0]), 2)
#y = np.arange(1, len(U[1]), 2)
#Ux = np.zeros((len(q.t), len(U[0])//2))
#Uy = np.zeros((len(q.t),len(U[0])//2)) 
#for t in range(len(q.t)):
    #U[t,:] = U_F[t,mask2]+NodeCoord[1:-1].flatten()
    ##print(U[t, :])
    #Ux[t] = U[t, 0::2]
    #Uy[t] = U[t, 1::2]
#
#
## add BC:
#BC1x = np.ones(len(q.t))*Top_cable
#BC1y = np.zeros(len(q.t))
#BC2x = np.ones(len(q.t))*Base_cable
#BC2y = np.zeros(len(q.t))
#Ux = np.insert(Ux, 0, BC1x, axis=1)
#Uy = np.insert(Uy, 0, BC1y, axis=1)
#Ux = np.insert(Ux, N_nodes-1, BC2x, axis=1)
#Uy = np.insert(Uy, N_nodes-1, BC2y, axis=1)
#plt.figure()
#plt.plot(Ux[20], Uy[20])
#
#fig = plt.figure()
#
#def animate(i):
    #plt.cla()
    #plt.title(f"Time is {q.t[i*100]:.1f} s")
    #plt.plot(Ux[i*100], Uy[i*100])
    #plt.ylim(-1, 1)
#
#anim = FuncAnimation(fig, animate, frames=len(q.t)//100,interval=1, repeat=False)
#
plt.show()
