# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:45:55 2022

@author: thijs
"""
# start of FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
from matplotlib.animation import FuncAnimation
from BeamMatrices import BeamMatricesJacket
from StringForcesAndStiffness import StringForcesAndStiffness
from analytical import *

class ContinousBeam(object):

    """Class that allows numerical simulation of a simple Euler-Bernoulli beam"""

    def __init__(self):
        """Sets up basic parameters and matrices"""
        # analytical solution
        self.analytical_omega = wn

        # Input parameters
        self.R = 0.5                       # [m] radius of pipe
        self.thickness = 0.010              # [m] wall thickness
        
        # cross-sectional properites
        self.A = 2*np.pi*self.R*self.thickness
        self.I = np.pi/2*(self.R**4-(self.R-self.thickness)**4)
    
        # properties of steel
        self.rho_cable = 7850               # [kg/m]
        self.E = 210e9                      # [N/m^2] Elasticity modulus of steel
        # Cable
                         # [kg/m]
        self.EI_pipe = self.E * self.I      # [N.m2]
        #self.EA_pipe = 0                   # neglecting elongation
        self.EA_pipe = self.E * self.A      # [N]
        self.top_cable = 0                  # [m]
        self.base_cable = 5000              # [m]

        # Define load parameters
        self.mode_force = 1
        self.f0 = 1                          # [Hz]
        self.A0 = 0.1                        # [m]
        self.T0 = 500000                        # [s]

        # Define output time vector
        self.dt = 0.01                       # [s]

        # Define node coordinates
        self.N_nodes = 100
        self.n_modes = 10


        # Modal analysis
        self.plot_n_modes = 5
        self.modal_damping_ratio = 0.05
        
        self.setup_system()
    
    def setup_system(self, convergence_run=False):
        ### SETUP SECONDARY PARAMETERS AND MATRICES
        # Setup time parameters
        self.T = np.arange(0, self.T0, self.dt)
        self.nT = len(self.T)

        # setup node coordinates 
        self.NodeCoord = np.zeros((self.N_nodes, 2))
        self.NodeCoord[:, 0] = np.linspace(self.top_cable, self.base_cable, self.N_nodes)

        # assign DOF's to nodes
        self.nDof = 3*self.N_nodes              # 3 Degrees of freedom per node
    
        # Boundary conditions
        # describe which DOF's are free and which are prescribed through the boundary conditions          # prescribed DOFs
        self.prescribed_dofs = np.array([0,  1, self.nDof-3, self.nDof-2])
        self.free_dofs = np.arange(0, self.nDof)       #  Initially set all DOF's as free DOf's
        self.free_dofs = np.delete(self.free_dofs, self.prescribed_dofs)  # remove the fixed DOFs from the free DOFs array

        # Numerical parameters
        self.max_modes = self.N_nodes*3 - len(self.prescribed_dofs)
        if not self.n_modes:
            self.n_modes = self.max_modes
        # Initial condition
        self.q0 = np.zeros(2*self.n_modes)

        # Setup matrices
        self.discretize_beam()
        self.setup_system_matrices()
        self.apply_boundary_conditions()
        self.setup_modal_matrix()
        if not convergence_run:
            self.setup_decoupled_system()
    
    def append_equation_to_latex(self, eq, fname="eoms_latex.txt"):
        with open(fname, "a") as f:
            f.writelines([sp.latex(eq, itex=True, mode="equation", mul_symbol="dot"), "\n\n\n"])

    def discretize_beam(self):
        # Define elements (and their properties
        #             NodeLeft    NodeRight          m         EA        EI
        self.N_elements = self.N_nodes - 1
        self.dx = (self.base_cable-self.top_cable)/self.N_elements
        self.m_pipe = self.rho_cable*self.dx*self.A
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
        #TODO: implement connectivity matrix
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
            
            mode_str = str(iMode-self.first_mode+1)
            title = f"Mode {mode_str}: f = {self.eigen_frequencies[iMode]:.3} Hz"
            plt.title(title)

        # automatically fix subplot spacing
        plt.tight_layout()
        plt.show()
    
    def plot_single_mode(self, iMode):
        if iMode < self.first_mode and iMode >= 0:
            print("WARNING -- This mode will not show anything") 
        
        plt.figure()
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
        if iMode >= 0:
            mode_str = str(iMode - self.first_mode + 1)
        else:
            mode_str = str(len(shape) + iMode + 1)

        title = f"Mode {mode_str}: f = {self.eigen_frequencies[iMode]:.3} Hz"
        plt.title(title)

    def setup_decoupled_system(self):
        # TODO: make number of modes dynamic
        # Compute Modal quantities and decouple the system
        self.contributing_modes = slice(self.first_mode, min(self.n_modes+self.first_mode, self.max_modes))
        self.PHI = self.eigen_vectors[:,self.contributing_modes]
       

        available_modes = self.contributing_modes.stop - self.contributing_modes.start
        self.Mm = np.zeros(available_modes)
        self.Km = np.zeros(available_modes)
        self.Cm = np.zeros(available_modes)

        # Compute your "self.n_modes" entries of the modal mass, stiffness and damping
        for i in range(available_modes):
            self.Mm[i] = self.PHI[:, i].T  @ self.M_FF @ self.PHI[:, i]
            self.Km[i] = self.PHI[:, i].T  @ self.K_FF @ self.PHI[:, i]
            self.Cm[i] = 2*self.modal_damping_ratio*np.sqrt(self.Mm[i]*self.Km[i])
            print(f"Computing Mode: {i+1} \nMm = {self.Mm[i]:.3f}, Km = {self.Km[i]:.3f}, Cm = {self.Cm[i]:.3f}")

    def get_forcing(self, t):
        F = np.zeros(len(self.free_dofs))
        f = 0.05* np.sin(np.arange(self.top_cable+self.dx, self.base_cable, self.dx)*np.pi*self.mode_force/(self.base_cable))
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
    def qdot(self, t,q):
        Um = q[0:self.n_modes]                                 # first entries are the displacemnt
        Vm = q[self.n_modes:2*self.n_modes]                           # second half is the velocity
        Am = ( self.get_forcing(t) - (self.Km * Um + self.Cm * Vm) ) / self.Mm      # compute the accelerations
        return np.append(Vm,Am)                         # return the velocity and accelerations

    # solving in time domain
    def get_solution_FEM(self):
        
        self.solution = scipy.integrate.solve_ivp(fun=self.qdot,y0=self.q0,t_span=[self.T[0],self.T[-1]])
        
        self.computed_time = self.solution.t
        print(self.get_forcing(self.solution.t).shape)
        self.accelerations = (((self.PHI @ self.get_forcing(self.solution.t))-(1 * self.solution.y[:self.n_modes, :]+ 1 *  self.solution.y[self.n_modes:, :])))/1
        
        #TODO: return to regular domain
        # displacement solutions from modal to spatial domain
        self.displacements = (self.PHI @ self.solution.y[:self.n_modes, :])
        
        # velocity solution from modal to spatial domain
        self.velocities = (self.PHI @ self.solution.y[self.n_modes:, :])

        print('Solving done')
    
    def get_support_reaction(self):
        # TODO: Document on translation/rotation matrix
        self.displacements[0]
    
    
    
    def get_frequency_convergence(self, dx_range_start, dx_range_stop):
        mode_list = [1,2,3,4,5]
        analytical_freq = [self.analytical_omega(n)/2/np.pi for n in mode_list]
    
        # 20 - 50
        dx_list = np.arange(dx_range_start, dx_range_stop)
        errors = np.zeros((len(dx_list), len(analytical_freq)))
        index = np.zeros(len(dx_list))
        for j,dx in enumerate(dx_list):
            self.N_nodes = int((self.base_cable-self.top_cable)/dx) + 1
            self.setup_system(convergence_run=True)
            numerical_freq = np.array([self.eigen_frequencies[i-1+self.first_mode] for i in mode_list])
            errors[j] = np.abs(analytical_freq - numerical_freq)
            index[j] = self.dx

        self.convergence_errors = errors
    
    def plot_frequency_convergence(self):
        df = pd.DataFrame(self.convergence_errors, columns=[1,2,3,4,5], index=np.arange(5,100))
        fig, ax = plt.subplots()
        ax.set_xlabel('dx [m]')
        ax.set_ylabel('Error [1/s]')
        ax.set_title('Modal frequency error for the first five modes')
        df.plot(ax=ax)
        ax.grid(True, which='both')
        ax.set_yscale('log')
        ax.set_xscale('log')
        
        
        
class Cable(object):
    
    def __init__(self):
        
        # physical parameters
        self.g = 9.81               # [m/s^2] gravity constant
        self.ksoil = 0       # [N/m] spring stiffnes soil
        self.soilpower = 1
        # material properties
        self.rho = 7850             # [kg/m^3] density of steel
        self.E = 210e9              # [N/m^2] elasticity modulus of steel
        
        # cross-sectional properties
        self.d = 0.05               # [m] diameter of cable
        self.A = np.pi*(self.d/2)**2 # [m^2] cross-sectional area    
        self.EA = self.E *self.A    # [Pa] stiffness

        # Initial position
        self.SAG = 10                            # [m] Choose a big enough sag to have as much elements under tension as possible
        self.anchor_depth = 1                   # [m] depth ofanchor below soil
        self.H = 20                                 # [m] height of pipeline above bed
        self.Fd = 0                           # [N] Horizontal design load 
        self.R = 15                                 # [m] anchor radius or distance between anchor and pipeline
        self.L = 2.3 * np.sqrt((self.H+self.anchor_depth)**2+self.R**2)   # [m] combined length of mooring lines
        
        # numerical parameters
        self.lmax = 1                               # [m] maximum length of each string(wire) element
        self.nElem = int(np.ceil(self.L/self.lmax)) # [-] number of elements  
        if self.nElem % 2 != 0:
            self.nElem +=1
        self.lElem = self.L/self.nElem              # [m] actual tensionless element size
        self.nNode = self.nElem + 1                 # [-] number of nodes 

        self.nMaxIter = 500                         # [-] max iterations done by solver
        self.TENSION_ONLY = 1                       # 0 both compression and tension, 1 tension only
       


        self.setup_system()

    def setup_system(self):
        self.discretize_cable()
        self.boundary_conditions()
        self.initial_position()
        self.define_external_forcing()
        
    
    def discretize_cable(self):
        self.NodeCoord = np.zeros((self.nNode, 2))
        self.Element = np.zeros((self.nElem, 5))
        
        for iElem in np.arange(0, self.nElem):
            NodeLeft = iElem
            NodeRight = iElem + 1
            self.NodeCoord[NodeRight] = self.NodeCoord[NodeLeft] + [self.lElem, 0]
            self.Element[iElem, :] = [NodeLeft, NodeRight, self.rho*self.A*self.lElem, self.EA, self.lElem]

    def plot_supports(self):
        plt.plot([-self.R, 0, self.R], [-self.anchor_depth, self.H, -self.anchor_depth], 'vr')
        
    def plot_discretization(self):
        plt.figure()
        for iElem in np.arange(0, self.nElem):
            NodeLeft = int(self.Element[iElem, 0])
            NodeRight = int(self.Element[iElem, 1])
        
            plt.plot([self.NodeCoord[NodeLeft][0], self.NodeCoord[NodeRight][0]], [self.NodeCoord[NodeLeft][1], self.NodeCoord[NodeRight][1]], 'g')
            
        # plot the supports
        self.plot_supports()
        plt.axis('equal')
        
    def boundary_conditions(self):
        self.nDof = 2*self.nNode                                    # number of DOFs
        self.FreeDof = np.arange(0, self.nDof)                      # free DOFs 
        self.FixedDof = [0,1,self.nDof//2,  -2,-1]               # fixed DOFs, vertical free but horizontal stiff
        self.FreeDof = np.delete(self.FreeDof, self.FixedDof)       # remove the fixed DOFs from the free DOFs array
        
        # free & fixed array indices
        self.fx = self.FreeDof[:, np.newaxis]
        self.fy = self.FreeDof[np.newaxis, :]

    def initial_position(self):
        # compute parabola for inital configuration
        self.u = np.zeros((self.nDof))
        # split the nodes in two mooring lines
        split = np.split(self.NodeCoord, [len(self.NodeCoord)//2+1, len(self.NodeCoord)-(len(self.NodeCoord)//2)])
        self.NodeCoordL, self.NodeCoordR = split[0], split[2]
        

        # left initial parabola
        x = np.array([-self.R, -self.R/2, 0])
        y = np.array([-self.anchor_depth, (self.anchor_depth+self.H)/2-self.SAG-self.anchor_depth,  self.H])
        A =np.zeros((3,3))
        for i, val in enumerate(x):
            A[i] = np.array([1,val, val**2])
        c, b, a =np.linalg.solve(A, y)
        # array of left Nodes
        s = np.array([i[0] for i in self.NodeCoordL])

        x = s*abs(-self.R + s[0])/abs(s[-1]-s[0]) -self.R
        y = a*x**2+b*x+c
        
        self.u[0:self.nDof//2:2]   = x - np.array([i[0] for i in self.NodeCoordL])
        self.u[1:self.nDof//2+1:2] = y - np.array([i[1] for i in self.NodeCoordL])

        # right initial parabola
        s = np.array([i[0] for i in self.NodeCoordR])

        x = np.array([0, self.R/2, self.R])
        y = np.array([self.H, (self.H+self.anchor_depth)/2-self.anchor_depth-self.SAG, -self.anchor_depth])
        
        A =np.zeros((3,3))
        for i, val in enumerate(x):
            A[i] = np.array([1,val, val**2])
        c, b, a =np.linalg.solve(A, y)
        
        
        
        x = self.R*((s-s[0])/(s[-1]-s[0]))
        y = a*x**2+b*x+c
        self.u[self.nDof//2+1::2] = x - np.array([i[0] for i in self.NodeCoordR])
        self.u[self.nDof//2+2::2] = y - np.array([i[1] for i in self.NodeCoordR])

    def plot_cable(self, Iter = None, label = ""):
        # plot the initial guess
        plt.figure()
        for iElem in np.arange(0, self.nElem):
            NodeLeft = int(self.Element[iElem, 0])
            NodeRight = int(self.Element[iElem, 1])
            DofsLeft = 2*NodeLeft 
            DofsRight = 2*NodeRight
            plt.plot([self.NodeCoord[NodeLeft][0] + self.u[DofsLeft], self.NodeCoord[NodeRight][0] + self.u[DofsRight]], 
                        [self.NodeCoord[NodeLeft][1] + self.u[DofsLeft + 1], self.NodeCoord[NodeRight][1] + self.u[DofsRight + 1]], '-k')
        
                
        # plot the supports
        self.plot_supports() 
        plt.hlines(0, -self.R*1.1, self.R*1.1, color= 'brown')
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        if Iter:
            plt.title("Iteration: "+ str(Iter))
        
    def define_external_forcing(self):
        self.Pext = np.zeros((self.nDof))
        for iElem in np.arange(0, self.nElem):
            NodeLeft = int(self.Element[iElem, 0])
            NodeRight = int(self.Element[iElem, 1])
            DofsLeft = 2*NodeLeft 
            DofsRight = 2*NodeRight
            l0 = self.Element[iElem, 4]
            m = self.Element[iElem, 2]
            Pelem = -self.g*l0*m/2                           # Half weight to each node
            self.Pext[DofsLeft + 1] += Pelem
            self.Pext[DofsRight + 1] += Pelem
        # external reaction force of pipe:
        self.Pext[self.nDof//2+1] = self.Fd
            
    def static_solver(self):
        CONV = 0
        PLOT = False
        self.kIter = 0
        self.TENSION = np.zeros((self.nElem))
        
        while CONV == 0:
            self.kIter += 1
            # if self.kIter % 100 ==0:
            #     print("Iteration: "+str(self.kIter)+" ...\n")
            # Check stability - define a number of maximum iterations. If solution
            # hasn't converged, check what is going wrong (if something).
            if self.kIter > self.nMaxIter:
                break
            
            # Assemble vector with internal forces and stiffnes matrix
            K = np.zeros((self.nDof*self.nDof)) 
            Fi = np.zeros((self.nDof))
            Fsoil = np.zeros((self.nDof))
            self.pos = self.NodeCoord[:, 0] + self.u[::2]
            self.pos = self.NodeCoord[:, 1] + self.u[1::2]
            
            
            for iElem in np.arange(0, self.nElem):
                NodeLeft = int(self.Element[iElem, 0])
                NodeRight = int(self.Element[iElem, 1])
                DofsLeft = 2*NodeLeft 
                DofsRight = 2*NodeRight
                l0 = self.Element[iElem, 4]
                EA = self.Element[iElem, 3]
                NodePos = ([self.NodeCoord[NodeLeft][0] + self.u[DofsLeft], self.NodeCoord[NodeRight][0] + self.u[DofsRight]], 
                            [self.NodeCoord[NodeLeft][1] + self.u[DofsLeft + 1], self.NodeCoord[NodeRight][1] + self.u[DofsRight + 1]])
                if NodePos[1][0] < 0 :
                    Fsoil[DofsLeft+1] = -(NodePos[1][0])**self.soilpower * self.ksoil
                if NodePos[1][1] < 0 and Fsoil[DofsRight] == 0:
                    Fsoil[DofsRight+1] = -(NodePos[1][1])**self.soilpower * self.ksoil
                Fi_elem, K_elem, Tension, WARN = StringForcesAndStiffness(NodePos, EA, l0, self.TENSION_ONLY)
                self.TENSION[iElem] = Tension
        
        
                # if WARN:
                #     print("WARNING: Element "+str(iElem+1)+" is under compression.\n")
                
                Fi[DofsLeft:DofsLeft + 2] += Fi_elem[0]
                Fi[DofsRight:DofsRight + 2] += Fi_elem[1]
        
                # Assemble the matrices at the correct place
                # Get the degrees of freedom that correspond to each node
                Dofs_Left = 2*(NodeLeft) + np.arange(0, 2)
                Dofs_Right = 2*(NodeRight) + np.arange(0, 2)
                nodes = np.append(Dofs_Left , Dofs_Right)
                for i in np.arange(0, 4):
                    for j in np.arange(0, 4):
                        ij = nodes[i] + nodes[j]*self.nDof
                        K[ij] = K[ij] + K_elem[i, j]
        
            K = K.reshape((self.nDof, self.nDof))
            
            
            # Calculate residual forces
            
            R = self.Pext+Fsoil - Fi
        
            # Check for convergence
            # only consider nodes above the ground for converges
            #cons_n = self.FreeDof[np.where(self.pos >0)]
            cons_n = self.FreeDof
            if np.linalg.norm(R[cons_n])/np.linalg.norm((self.Pext[cons_n]+Fsoil[cons_n])) < 1e-1:
                CONV = 1
        
            # Calculate increment of displacements
            du = np.zeros((self.nDof))
            du[self.FreeDof] = np.linalg.solve(K[self.fx, self.fy], R[self.FreeDof])
        
            # Apply archlength to help with convergence
            Scale = np.min(np.append(np.array([1]), self.lElem/np.max(np.abs(du))))
            du = du*Scale   # Enforce that each node does not displace
                            # more (at each iteration) than the length
                            # of the elements
        
            # Update displacement of nodes
            self.u += du
            
            # plot the updated configuration
            if PLOT :
                print(Fsoil)
                self.plot_cable(Iter = self.kIter)
        
        if CONV == 1:
            print("Converged solution at iteration: "+str(self.kIter))
            #self.plot_cable(Iter = self.kIter)
            # store solution
            self.store_solution()
            
        else:
            print("Solution did not converge")
            self.store_solution()
        
    def store_solution(self):
        self.sol = np.zeros((self.NodeCoord.T.shape))
        self.sol[0] = self.NodeCoord.T[0] + self.u[::2]
        self.sol[1] = self.NodeCoord.T[1] + self.u[1::2]
    
    def plot_tension(self):
        plt.figure()
        X = (np.array([i[0] for i in self.NodeCoord[0:-1]]) + np.array([i[0] for i in self.NodeCoord[1:]]))/2
        print(X)
        plt.plot(X, self.TENSION)
        plt.title("Tension")
        plt.xlabel("x [m]")
        plt.ylabel("T [N]")
    