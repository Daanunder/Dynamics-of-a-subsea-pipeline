# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:45:55 2022

@author: thijs
"""
import sys
sys.path.insert(1, '../part_2')
# start of FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
from matplotlib.animation import FuncAnimation
from BeamMatrices import BeamMatricesJacket
from analytical import *
from forcing import get_forcing as get_irregular_forcing

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
        self.EA_pipe = self.E * self.A      # [N]
        self.top_cable = 0                  # [m]
        self.base_cable = 5000              # [m]

        # Define load parameters
        self.f0 = 2                          # [Hz]
        self.A0 = 0.1                        # [m]
        self.T0 = 100000                         # [s]
        
        # irregular forcing parameters
        self.apply_irregular_forcing = True
        self.Hm0 = 4 #m
        self.Tp = 20 #s
        self.CD = 0.7 #-
        self.ship_mass = 60e6 #kg
        self.ship_area = 12000  #m2
        self.rho_water = 1000 #kg/m3

        # Define output time vector
        self.dt = 1                       # [s]

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

        # Irregular forcing function
        self.omega_waves = np.pi*2/self.Tp
        self.amplitude_time_series = get_irregular_forcing(self.Hm0, self.Tp, self.T)[0]
        self.irregular_forcing = 1/2*self.rho_water*(self.omega_waves*self.amplitude_time_series)**2*self.CD*self.ship_area


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
            Dofs_Left  = 3*(NodeLeft)  + np.arange(0, 3)
            Dofs_Right = 3*(NodeRight) + np.arange(0, 3)

            # Get the properties of the element
            m  = element[2]
            EA = element[3]
            EI = element[4]

            # Calculate the local matrix of each element
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
        self.M_FP = M_FP

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
        print("System Characteristics :")
        for i in range(available_modes):
            self.Mm[i] = self.PHI[:, i].T  @ self.M_FF @ self.PHI[:, i]
            self.Km[i] = self.PHI[:, i].T  @ self.K_FF @ self.PHI[:, i]
            self.Cm[i] = 2*self.modal_damping_ratio*np.sqrt(self.Mm[i]*self.Km[i])
            print(f"Mode: {i+1}: \t\t Mm = {self.Mm[i]:.3f}, Km = {self.Km[i]:.3f}, Cm = {self.Cm[i]:.3f}")
        print("---------------------------\n")

    def get_forcing(self, t):
        if not self.apply_irregular_forcing:
            c_F = 20 #kN
            F = np.zeros(len(self.free_dofs))
            f = 0.8* np.sin(np.arange(self.top_cable+self.dx, self.base_cable, self.dx)*np.pi/self.base_cable)
            F[2::3] = f 
            Fequ  = np.dot(self.PHI.T, F)
            return Fequ
        else:
            # prescribe acceleration by dividing force over ship mass
            A = np.zeros(len(self.prescribed_dofs))
            force_index = np.argwhere(self.T >= t)[0][0]
            A[2] = self.irregular_forcing[force_index] * 1000
            F =  - self.M_FP.T @ A 
            return self.PHI.T @ F

    # update function
    def qdot(self, t, q):
        if np.round(t % 1000) < 5 and t > 1000:
            print("Time :",t, "Percentage :" , np.round(t/self.T[-1] * 100), end="\r")
        Um = q[0:self.n_modes]                                 # first entries are the displacemnt
        Vm = q[self.n_modes:2*self.n_modes]                           # second half is the velocity
        Am = ( self.get_forcing(t) - (self.Km * Um + self.Cm * Vm) ) / self.Mm      # compute the accelerations
        return np.append(Vm,Am)                         # return the velocity and accelerations

    # solving in time domain
    def get_solution_FEM(self):
        
        self.solution = scipy.integrate.solve_ivp(fun=self.qdot,y0=self.q0,t_span=[self.T[0],self.T[-1]])

        self.computed_time = self.solution.t
        
        # displacement solutions from modal to spatial domain
        self.displacements = np.array([self.PHI @ self.solution.y[:self.n_modes, t] for t in range(len(self.computed_time))]).T
        zero_row = np.zeros(self.displacements.shape[1])
        for pdof in self.prescribed_dofs:
            p1 = self.displacements[:pdof]
            p2 = self.displacements[pdof:]
            self.displacements = np.concatenate((np.concatenate((p1, [zero_row]), axis=0), p2), axis=0)

        self.x_displacements = self.displacements[0::3]
        self.y_displacements = self.displacements[1::3]
        self.theta_displacements = self.displacements[2::3]
        self.curvature_displacements = np.abs(np.diff(self.theta_displacements, axis=0))*np.pi/180/(2*self.dx)
        
        # velocity solution from modal to spatial domain
        self.velocities = np.array([self.PHI @ self.solution.y[self.n_modes:, t] for t in range(len(self.computed_time))]).T
        self.x_velocities = self.velocities[0::3]
        self.y_velocities = self.velocities[1::3]
        self.theta_velocities = self.velocities[2::3]
    
        print()
        print('Solving done')
        
    def plot_maximum_beam_displacement(self, fig=None, ax=None, ls="-"):
        save = False
        if not fig:
            save = True
            fig, ax = plt.subplots(2,2, figsize=(16, 12))
        
        ## Assume steady state is beyond half of the computation time
        mask = self.computed_time > self.T[-1]/2
        extra_index = len(mask) - np.count_nonzero(mask)

        curvature_timestep = max(self.curvature_displacements[:,mask].argmax(axis=1)) + extra_index
        y_timestep = max(self.y_displacements[:,mask].argmax(axis=1)) + extra_index

        curvature_node = max(self.curvature_displacements[:,mask].argmax(axis=0))
        y_node = max(self.y_displacements[:,mask].argmax(axis=0))

        curvature1 = self.curvature_displacements[:, curvature_timestep]
        y1 = self.y_displacements[:, y_timestep]

        curvature2 = self.curvature_displacements[curvature_node, :]
        y2 = self.y_displacements[y_node, :]

        a = ax[0,0]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), curvature1, label=f"{self.Hm0} - {self.Tp}", ls=ls)
        handles, labels = a.get_legend_handles_labels()
        a.set_title(f"Maximum curvature occuring at t={np.round(self.computed_time[curvature_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Curvature [deg/m]')
        a.grid()

        a = ax[0,1]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes), y1, ls=ls)
        a.set_title(f"Maximum perpendicular displacement occuring at t={np.round(self.computed_time[y_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Displacement [m]')
        a.grid()

        a = ax[1,0]
        a.plot(self.computed_time, curvature2, ls=ls)
        a.set_title(f"Curvature occuring at element {curvature_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Curvature [deg/m]')
        a.grid()

        a = ax[1,1]
        a.plot(self.computed_time, y2, ls=ls)
        a.set_title(f"Perpendicular displacement occuring at element {y_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Displacement [m]')
        a.grid()
       
        if save:
            fig.suptitle(f"Maximum displacements and curvature for steady state (t > 500000s) and for single point over time")
            fig.legend(handles, labels, title="Hm0 - Tp")
            plt.savefig("system_response_irregular_forcing.png")
            plt.cla()
        return handles, labels

    def get_system_response_irregular_forcing(self, Hm0_range, Tp_range):
        fig, ax = plt.subplots(2,2, figsize=(16, 12))
        all_handles = []
        all_labels = []
        line_styles = ["-", "--", "-.", ":"]
        for Hm0 in Hm0_range:
            for i,Tp in enumerate(Tp_range):
                ls = line_styles[i]
                print("=================")
                print("Running simulation for Hm0: ", Hm0, " Tp: ", Tp)
                print("=================")
                self.Hm0 = float(Hm0)
                self.Tp = float(Tp)
                self.setup_system()
                self.get_solution_FEM()
                handles, labels = self.plot_maximum_beam_displacement(fig, ax, ls)
                all_handles.extend(handles)
                all_labels.extend(labels)

        fig.suptitle(f"Maximum displacements of the full beam when steady state is reached (> 500000s) and for single point over time")
        fig.legend(handles, labels, title="Hm0 - Tp")
        plt.savefig("system_response_irregular_forcing.png")
        plt.cla()

    
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


