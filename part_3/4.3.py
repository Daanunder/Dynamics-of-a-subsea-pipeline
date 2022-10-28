# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:11:54 2022

@author: thijs
"""

import numpy as np
from matplotlib import pyplot as plt
from StringForcesAndStiffness import StringForcesAndStiffness

class Cable(object):
    
    def __init__(self):
        
        # physical parameters
        self.g = 9.81               # [m/s^2] gravity constant
        self.ksoil = 430        # [N/m] spring stiffnes soil
        self.soilmodel = 1.2      # type of spring 1 linear 2 quadratic
        # material properties
        self.rho = 7850             # [kg/m^3] density of steel
        self.E = 210e9              # [N/m^2] elasticity modulus of steel
        
        # cross-sectional properties
        self.d = 0.05               # [m] diameter of cable
        self.A = np.pi*(self.d/2)**2 # [m^2] cross-sectional area    
        self.EA = self.E *self.A    # [Pa] stiffness

        # Initial position
        self.SAG = 15                            # [m] Choose a big enough sag to have as much elements under tension as possible
        self.anchor_depth = 1                   # [m] depth ofanchor below soil
        self.H = 20                                 # [m] height of pipeline above bed
        self.hor_offset = 0                         # horizontal offset vessel
        self.Fd = 1300                               # [N] Horizontal design load 
        self.R = 15                                 # [m] anchor radius or distance between anchor and pipeline
        self.L = 30  # [m] combined length of mooring lines
        
        # numerical parameters
        self.lmax = 1                               # [m] maximum length of each string(wire) element
        self.nElem = int(np.ceil(self.L/self.lmax)) # [-] number of elements  
        if self.nElem % 2 != 0:
            self.nElem +=1
        self.lElem = self.L/self.nElem              # [m] actual tensionless element size
        self.nNode = self.nElem + 1                 # [-] number of nodes 

        self.nMaxIter = 200                         # [-] max iterations done by solver
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
        plt.plot([-self.R, self.hor_offset], [-self.anchor_depth, self.H], 'vr')
        
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
        self.FixedDof = [0,1,-2,-1]                                 # fixed DOFs, vertical free but horizontal stiff
        self.FreeDof = np.delete(self.FreeDof, self.FixedDof)       # remove the fixed DOFs from the free DOFs array
        
        # free & fixed array indices
        self.fx = self.FreeDof[:, np.newaxis]
        self.fy = self.FreeDof[np.newaxis, :]

    def initial_position(self):
        # compute parabola for inital configuration
        self.u = np.zeros((self.nDof))
        # split the nodes in two mooring lines

        

        # left initial parabola
        x = np.array([-self.R, -self.R/2, self.hor_offset])
        y = np.array([-self.anchor_depth, (self.anchor_depth+self.H)/2-self.SAG-self.anchor_depth,  self.H])
        A =np.zeros((3,3))
        for i, val in enumerate(x):
            A[i] = np.array([1,val, val**2])
        c, b, a =np.linalg.solve(A, y)
        # array of left Nodes
        s = np.array([i[0] for i in self.NodeCoord])
        length = self.R+self.hor_offset
        x = s*length/abs(s[0]-s[-1]) -self.R
        y = a*x**2+b*x+c
        
        self.u[0::2]   = x - np.array([i[0] for i in self.NodeCoord])
        self.u[1::2] = y - np.array([i[1] for i in self.NodeCoord])

       
        

    def plot_cable(self, Iter = None, label = ""):
        # plot the initial guess
        
        for iElem in np.arange(0, self.nElem):
            NodeLeft = int(self.Element[iElem, 0])
            NodeRight = int(self.Element[iElem, 1])
            DofsLeft = 2*NodeLeft 
            DofsRight = 2*NodeRight
            plt.plot([self.NodeCoord[NodeLeft][0] + self.u[DofsLeft], self.NodeCoord[NodeRight][0] + self.u[DofsRight]], 
                        [self.NodeCoord[NodeLeft][1] + self.u[DofsLeft + 1], self.NodeCoord[NodeRight][1] + self.u[DofsRight + 1]], '-k')
        
                
        # plot the supports
        self.plot_supports() 
        plt.hlines(0, -self.R*1.1, self.R*0.5, color= 'brown')
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
        self.Pext[-2] = self.Fd
            
    def static_solver(self):
        CONV = 0
        PLOT = False
        self.kIter = 0
        self.TENSION = np.zeros((self.nElem))
        store_R = np.zeros(self.nMaxIter)
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
                    Fsoil[DofsLeft+1] = (-NodePos[1][0])**self.soilmodel * self.ksoil
                if NodePos[1][1] < 0 and Fsoil[DofsRight] == 0:
                    Fsoil[DofsRight+1] = (-NodePos[1][1])**self.soilmodel * self.ksoil
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
            cons_n = self.FreeDof[np.where(self.pos >0)]
            #cons_n = self.FreeDof
            store_R[self.kIter-1] = np.linalg.norm(R[cons_n])/np.linalg.norm((self.Pext[cons_n]+Fsoil[cons_n]))
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
                
                self.plot_cable(Iter = self.kIter)
        
        if CONV == 1:
            print("Converged solution at iteration: "+str(self.kIter))
            #self.plot_cable(Iter = self.kIter)
            # store solution
            self.store_solution()
            self.rest = np.min(store_R)
            
            
        else:
            self.rest = np.min(store_R)
            self.TENSION = [np.nan]
            print("Solution did not converge")
        
        
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
# ml = Cable()
# plt.figure(figsize=(10, 8))

# offset = np.arange(1,30, 1)
# for i, off in enumerate(offset):
#     ml = Cable()
#     ml.hor_offset = off
#     ml.setup_system()
#     ml.static_solver()
#     ml.plot_cable()

# plt.title("Positions analysed")

offset = np.arange(1,180, 2)
tension = np.zeros(len(offset))
for i, off in enumerate(offset):
    ml = Cable()
    ml.hor_offset = off
    ml.setup_system()
    ml.static_solver()
    tension[i] = ml.TENSION[-1]
print(tension)
xlabel = (np.sqrt(20**2+(15+offset)**2))/30
print(xlabel)


plt.figure(figsize =(12, 6))
diff = abs(tension - np.roll(tension, -1))/2
plt.hlines(ml.EA/ml.L, offset[0], offset[-1], linestyle='--', color = 'orange', label = "EA/L")

plt.plot(offset, diff, label = "Stiffness mooring line")
plt.ylabel('K [N/m]')
plt.title("Stiffnes mooring line")
plt.xlabel('offset [m]')
plt.legend(bbox_to_anchor=(0.55, 0.65),ncol=1, fancybox=True, shadow=True)
# plt.xlim(0.875, 0.9)
# #plt.savefig("tension,offset,s=0.5,0.8")

# plt.figure(figsize =(12, 6))
# plt.plot(xlabel[:], tension[:])
# plt.ylabel('T [N]')
# plt.yscale('log')
# plt.title("Tension in mooring line at connection point")
# plt.xlabel('s/L [-]')
# plt.savefig("tension,offset,s")

# plt.figure()

# xlabel = (np.sqrt(20**2+(15+offset)**2))/30
# plt.plot(xlabel,tension)
# plt.figure()

# plt.plot(offset,tension)
# plt.xscale('log')
# plt.yscale('log')
# plt.plot(offset, offset*20)
# diff = tension - np.roll(tension, -1)
# plt.figure()
# plt.plot(offset, diff)
# print(diff)
    
  
# ml = Cable()
# power = np.linspace(0, 1000, 50)
# R_min = np.zeros(50)
# for i in range(50):
#     ml.ksoil = power[i]
#     ml.static_solver()
#     R_min[i] = ml.rest
# plt.figure()
# plt.plot(power, R_min)