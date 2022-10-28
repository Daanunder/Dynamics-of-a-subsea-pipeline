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
import datetime
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
        self.thickness = 0.01              # [m] wall thickness
        
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
        self.T0 = 800000
        
        # irregular forcing parameters
        self.apply_irregular_forcing = True
        self.Hm0 = 6 #m
        self.Tp = 12 #s
        self.CD = 0.07 #-
        self.ship_mass = 60e6 #kg
        self.ship_area = 6000  #m2
        self.rho_water = 1000 #kg/m3
        #self.wave_characteristics = [(4,10), (4,12), (6, 12), (6, 14), (8, 12), (8, 14), (8, 16), (10, 14), (10, 16), (12, 16), (16, 18)]
        self.wave_characteristics = [(4,10),(6, 12)] #,(8, 14),(12, 16),(16, 18)]
        self.K_ship = 10e8

        # Define output time vector
        self.dt = 0.1                       # [s]

        # Define node coordinates
        self.N_nodes = 100
        self.n_modes = 10

        # Modal analysis
        self.plot_n_modes = 5
        self.modal_damping_ratio = 0.05
       
        # Setup secondary parameters
        self.setup_system()
        
    
    def setup_system(self, convergence_run=False):
        ### SETUP SECONDARY PARAMETERS AND MATRICES
        # Setup time parameters
        self.T = np.arange(0, self.T0, self.dt)
        self.forcing_t = np.arange(0, self.T0+self.Tp/4, self.dt)
        self.nT = len(self.T)

        # Irregular forcing function
        self.omega_waves = np.pi*2/self.Tp
        # make sure velocity and amplitude are 90deg out of phase
        phase_mask = self.forcing_t >= self.Tp/4
        self.amplitude_time_series = get_irregular_forcing(self.Hm0, self.Tp, self.forcing_t)[0][phase_mask]
        self.flow_velocity_time_series = 1/2*self.rho_water*(self.omega_waves*self.amplitude_time_series)**2*self.CD*self.ship_area
        self.irregular_forcing = [self.amplitude_time_series, self.flow_velocity_time_series]

        # Workability limits
        # steel yield strenght = 355
        self.max_curvature = 355*10**6 / (self.E * self.R)
        self.max_drift = 1000.0 
        self.max_strain = 355*10**6 / self.E
        self.max_bending_stress = 355*10**6
        self.max_normal_stress = 355*10**6
        self.max_total_stress = 355*10**6

        # workability plots
        self.plot_curvature_treshold = False
        self.plot_drift_treshold = False
        self.plot_strain_treshold = False
        self.plot_bending_stress_treshold = False
        self.plot_normal_stress_treshold = False
        self.plot_total_stress_treshold = False


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
        self.K_FP = K_FP

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
        # Compute Modal quantities and decouple the system
        self.contributing_modes = slice(self.first_mode, min(self.n_modes+self.first_mode, self.max_modes))
        self.PHI = self.eigen_vectors[:,self.contributing_modes]
       

        available_modes = self.contributing_modes.stop - self.contributing_modes.start
        self.Mm = np.zeros(available_modes)
        self.Km = np.zeros(available_modes)
        self.Cm = np.zeros(available_modes)

        # Compute your "self.n_modes" entries of the modal mass, stiffness and damping
        #print("System Characteristics :")
        for i in range(available_modes):
            self.Mm[i] = self.PHI[:, i].T  @ self.M_FF @ self.PHI[:, i]
            self.Km[i] = self.PHI[:, i].T  @ self.K_FF @ self.PHI[:, i]
            self.Cm[i] = 2*self.modal_damping_ratio*np.sqrt(self.Mm[i]*self.Km[i])
            #print(f"Mode: {i+1}: \t\t Mm = {self.Mm[i]:.3f}, Km = {self.Km[i]:.3f}, Cm = {self.Cm[i]:.3f}")
        #print("---------------------------\n")

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
            U = np.zeros(len(self.prescribed_dofs))
            UDD = np.zeros(len(self.prescribed_dofs))
            force_index = np.argwhere(self.T >= t)[0][0]
            U[0] = self.irregular_forcing[0][force_index] * self.K_ship
            UDD[1] = self.irregular_forcing[1][force_index] / self.ship_mass
            F = -self.K_FP.T @ U - self.M_FP.T @ UDD
            return self.PHI.T @ F

    # update function
    def qdot(self, t, q):
        if np.round(t % 1000) < 5 and t > 1000:
            p = t/self.T[-1]
            d = datetime.datetime.now() - self.simulation_starttime 
            r = d.total_seconds()/p
            time_to_go = datetime.timedelta(seconds=(1-p)*r)
            print("Time :",t, "\t Percentage :" , np.round(p*100) , "\t Estimated time left: " , time_to_go, end="\r")
        Um = q[0:self.n_modes]                                 # first entries are the displacemnt
        Vm = q[self.n_modes:2*self.n_modes]                           # second half is the velocity
        Am = ( self.get_forcing(t) - (self.Km * Um + self.Cm * Vm) ) / self.Mm      # compute the accelerations
        return np.append(Vm,Am)                         # return the velocity and accelerations

    # solving in time domain
    def get_solution_FEM(self):
        self.simulation_starttime = datetime.datetime.now()
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

        self.curvature = np.diff(self.theta_displacements, axis=0)*np.pi/180/(2*self.dx)
        self.strain = np.diff(self.x_displacements, axis=0)
        
        ## Stresses in N/m2
        self.normal_stress = self.strain*self.E /10**6
        self.bending_stress = self.curvature*self.E*self.R /10**6
        self.total_stress = self.bending_stress + self.normal_stress /10**6
        
        # velocity solution from modal to spatial domain
        self.velocities = np.array([self.PHI @ self.solution.y[self.n_modes:, t] for t in range(len(self.computed_time))]).T
        self.x_velocities = self.velocities[0::3]
        self.y_velocities = self.velocities[1::3]
        self.theta_velocities = self.velocities[2::3]
        
        duration = datetime.datetime.now() - self.simulation_starttime
        print()
        print(f'Solving done, took me: {duration} hh:mm:ss')
        
    def plot_maximum_beam_displacement(self, fig=None, ax=None, ls="-", label=None):
        ## Assume steady state is beyond half of the computation time
        mask = self.computed_time > self.T[-1]/2
        extra_index = len(mask) - np.count_nonzero(mask)

        self.plot_curvature_treshold = np.abs(self.curvature).max() > self.max_curvature if not self.plot_curvature_treshold else True
        self.plot_drift_treshold = np.abs(self.y_displacements).max() > self.max_drift if not self.plot_drift_treshold else True
        self.plot_strain_treshold = np.abs(self.strain).max() > self.max_strain if not self.plot_strain_treshold else True

        curvature_timestep = max(np.abs(self.curvature[:,mask]).argmax(axis=1)) + extra_index
        y_timestep = max(np.abs(self.y_displacements[:,mask]).argmax(axis=1)) + extra_index
        strain_timestep = max(np.abs(self.strain[:,mask]).argmax(axis=1)) + extra_index

        curvature_node = max(np.abs(self.curvature[:,mask]).argmax(axis=0))
        y_node = max(np.abs(self.y_displacements[:,mask]).argmax(axis=0))
        strain_node = max(np.abs(self.strain[:,mask]).argmax(axis=0))

        curvature1 = self.curvature[:, curvature_timestep]
        y1 = self.y_displacements[:, y_timestep]
        strain1 = self.strain[:, strain_timestep]

        curvature2 = self.curvature[curvature_node, :]
        y2 = self.y_displacements[y_node, :]
        strain2 = self.strain[strain_node, :]

        a = ax[0,0]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), curvature1, label=label, ls=ls)
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

        a = ax[0,2]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), strain1, ls=ls)
        a.set_title(f"Maximum strain occuring at t={np.round(self.computed_time[strain_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Strain [m/m]')
        a.grid()

        a = ax[1,0]
        a.plot(self.computed_time, curvature2, ls=ls)
        a.set_title(f"Curvature occuring at element {curvature_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Curvature [1/m]')
        a.grid()

        a = ax[1,1]
        a.plot(self.computed_time, y2, ls=ls)
        a.set_title(f"Perpendicular displacement occuring at element {y_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Displacement [m]')
        a.grid()

        a = ax[1,2]
        a.plot(self.computed_time, strain2, ls=ls)
        a.set_title(f"Strain occuring at element {strain_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Strain [m/m]')
        a.grid()

    def plot_stresses(self, fig, ax, ls="-", label=None):
        ## Assume steady state is beyond half of the computation time
        mask = self.computed_time > self.T[-1]/2
        extra_index = len(mask) - np.count_nonzero(mask)

        self.plot_bending_stress_treshold = np.abs(self.bending_stress).max() > self.max_bending_stress if not self.plot_bending_stress_treshold else True
        self.plot_normal_stress_treshold = np.abs(self.normal_stress).max() > self.max_normal_stress if not self.plot_normal_stress_treshold else True
        self.plot_total_stress_treshold = np.abs(self.total_stress).max() > self.max_total_stress if not self.plot_total_stress_treshold else True

        bending_stress_timestep = max(np.abs(self.bending_stress[:,mask]).argmax(axis=1)) + extra_index
        normal_stress_timestep = max(np.abs(self.normal_stress[:,mask]).argmax(axis=1)) + extra_index
        total_stress_timestep = max(np.abs(self.total_stress[:,mask]).argmax(axis=1)) + extra_index

        bending_stress_node = max(np.abs(self.bending_stress[:,mask]).argmax(axis=0))
        normal_stress_node = max(np.abs(self.normal_stress[:,mask]).argmax(axis=0))
        total_stress_node = max(np.abs(self.total_stress[:,mask]).argmax(axis=0))

        bending_stress1 = self.bending_stress[:, bending_stress_timestep]
        normal_stress1 = self.normal_stress[:, normal_stress_timestep]
        total_stress1 = self.total_stress[:, total_stress_timestep]

        bending_stress2 = self.bending_stress[bending_stress_node, :]
        normal_stress2 = self.normal_stress[normal_stress_node, :]
        total_stress2 = self.total_stress[total_stress_node, :]

        a = ax[0,0]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), bending_stress1, label=label, ls=ls)
        a.set_title(f"Maximum bending stress occuring at t={np.round(self.computed_time[bending_stress_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Bending stress [Pa]')
        a.grid()
        a = ax[0,1]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), normal_stress1, ls=ls)
        a.set_title(f"Maximum normal stress occuring at t={np.round(self.computed_time[normal_stress_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Normal stress [Pa]')
        a.grid()
        a = ax[0,2]
        a.plot(np.linspace(self.top_cable, self.base_cable, self.N_nodes-1), total_stress1, ls=ls)
        a.set_title(f"Maximum total stress occuring at t={np.round(self.computed_time[total_stress_timestep],2)}s")
        a.set_xlabel('Beam length [m]')
        a.set_ylabel('Total stress [Pa]')
        a.grid()


        a = ax[1,0]
        a.plot(self.computed_time, bending_stress2, ls=ls)
        a.set_title(f"Bending stress occuring at element {bending_stress_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Bending stress [Pa]')
        a.grid()
        a = ax[1,1]
        a.plot(self.computed_time, normal_stress2, ls=ls)
        a.set_title(f"Normal stress occuring at element {normal_stress_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Normal stress [Pa]')
        a.grid()
        a = ax[1,2]
        a.plot(self.computed_time, total_stress2, ls=ls)
        a.set_title(f"Total stress occuring at element {total_stress_node} over time")
        a.set_xlabel('Time [s]')
        a.set_ylabel('Total stress [Pa]')
        a.grid()

    def get_system_response_irregular_forcing(self, wave_characteristics=None, renew_figure=False):
        if not wave_characteristics:
            wave_characteristics = self.wave_characteristics
        
        if renew_figure:
            self.fig, self.ax = plt.subplots(2,3, figsize=(20, 14))
            self.fig2, self.ax2 = plt.subplots(2,3, figsize=(20, 14))

        line_styles = ["-", "--", "-.", ":"]
        for i, (Hm0, Tp) in enumerate(wave_characteristics):
            ls = line_styles[i % 4]
            print("=================")
            print("Running simulation for Hm0: ", Hm0, " Tp: ", Tp)
            self.Hm0 = float(Hm0)
            self.Tp = float(Tp)
            self.setup_system()
            self.get_solution_FEM()
            label = f"{self.Hm0} - {self.Tp}"
            self.plot_maximum_beam_displacement(self.fig, self.ax, ls, label= label)
            self.plot_stresses(self.fig2, self.ax2, ls)
            
            print("=================")

        if self.plot_curvature_treshold:
            self.ax[0,0].hlines([self.mself.ax_curvature, -self.mself.ax_curvature], self.base_cable, self.top_cable)
            self.ax[1,0].hlines([self.mself.ax_curvature, -self.mself.ax_curvature], self.T[0], self.T[-1])
        if self.plot_drift_treshold:
            self.ax[0,1].hlines([self.mself.ax_drift, -self.mself.ax_drift], self.base_cable, self.top_cable)
            self.ax[1,1].hlines([self.mself.ax_drift, -self.mself.ax_drift], self.T[0], self.T[-1])
        if self.plot_strain_treshold:
            self.ax[0,2].hlines([self.mself.ax_strain, -self.mself.ax_strain], self.base_cable, self.top_cable)
            self.ax[1,2].hlines([self.mself.ax_strain, -self.mself.ax_strain], self.T[0], self.T[-1])

        if self.plot_bending_stress_treshold:
            self.ax2[0,0].hlines([self.mself.ax_bending_stress, -self.mself.ax_bending_stress], self.base_cable, self.top_cable)
            self.ax2[1,0].hlines([self.mself.ax_bending_stress, -self.mself.ax_bending_stress], self.T[0], self.T[-1])
        if self.plot_normal_stress_treshold:
            self.ax2[0,1].hlines([self.mself.ax_normal_stress, -self.mself.ax_normal_stress], self.base_cable, self.top_cable)
            self.ax2[1,1].hlines([self.mself.ax_normal_stress, -self.mself.ax_normal_stress], self.T[0], self.T[-1])
        if self.plot_total_stress_treshold:
            self.ax2[0,0].hlines([self.mself.ax_total_stress, -self.mself.ax_total_stress], self.base_cable, self.top_cable)
            self.ax2[1,0].hlines([self.mself.ax_total_stress, -self.mself.ax_total_stress], self.T[0], self.T[-1])

        handles, labels = self.ax[0,0].get_legend_handles_labels()

        self.fig.suptitle(f"Workability for the steady state (t > 500000s), with maximum curvature: {self.max_curvature:.2}, maximum perpendicular drift: {self.max_drift:.2} and maximum strain {self.max_strain:.2}", fontsize=16)
        self.fig.legend(handles, labels, title="Hm0 - Tp")
        #plt.tight_layout()
        self.fig.savefig("system_response_irregular_forcing.png")

        self.fig2.suptitle(f"Stress occuring for different loading conditions, where the maximum stress is defined as 355 Pa", fontsize=16)
        self.fig2.legend(handles, labels, title="Hm0 - Tp")
        #plt.tight_layout()
        self.fig2.savefig("stress_response_irregular_forcing.png")
   
    def get_spatial_convergence(self):
        phase_mask = self.forcing_t >= self.Tp/4
        amplitude_time_series = get_irregular_forcing(self.Hm0, self.Tp, self.forcing_t)[0][phase_mask]
        flow_velocity_time_series = 1/2*self.rho_water*(self.omega_waves*self.amplitude_time_series)**2*self.CD*self.ship_area
        irregular_forcing = [self.amplitude_time_series, self.flow_velocity_time_series]
        #n_nodes_range = [100, 100, 200, 200, 300, 300, 400, 400]
        #n_modes_range = [10, 50, 20, 100, 30, 150, 40, 200]
        #n_nodes_range = np.arange(100, 1000, 100)
        #n_modes_range = [10]*len(n_nodes_range)
        n_nodes_range = [800, 900]*2
        n_modes_range = [80, 90, 10, 10]

        fig, ax = plt.subplots(2,3, figsize=(20, 14))
        
        for nodes, modes in zip(n_nodes_range, n_modes_range):
            print("=================")
            print(f"Running spatial convergence, Nodes: {nodes} - Modes: {modes}")
            self.N_nodes = nodes
            self.n_modes = modes
            self.setup_system()
            self.irregular_forcing = irregular_forcing
            self.get_solution_FEM()
            label = f'{nodes}-{modes}'
            self.plot_maximum_beam_displacement(fig, ax, label=label)
            print("=================")
        
        handles, labels = ax[0,0].get_legend_handles_labels()

        fig.suptitle(f"System response for different number of nodes and different number of included modes", fontsize=16)
        fig.legend(handles, labels, title="N_nodes - N_modes")
        #plt.tight_layout()
        fig.savefig("spatial_convergence_plot.png")
    
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


