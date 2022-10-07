import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.optimize import fsolve

import sympy as sp
from sympy.physics.mechanics import init_vprinting, dynamicsymbols
from sympy.utilities.lambdify import lambdify
from sympy.abc import t
from sympy import init_printing
init_vprinting()



class NDOF_system(object):

    """Docstring for NDOF_system. """

    def __init__(self, R=5):
        # Pipe and system parameters
        self.t_pipe = 0.004 # m
        self.L = 5000 #m  
        self.D_pipe = 0.5 #m
        self.rho_steel = 7850
        self.rho_water = 1000
        self.total_mass =  np.pi*self.D_pipe*self.t_pipe*self.L*self.rho_steel              # [kg]
        self.I = 1/2*self.total_mass/self.L*self.D_pipe**2
        self.E = 210000
        self.EI = self.E*self.I
        self.g_eff = (self.rho_steel - self.rho_water)/self.rho_steel * 9.81
        self.total_kx = 16*self.EI/(self.L/2)**2 + 2*self.total_mass*self.g_eff/self.L    # [N/m] Structural Bending stifness + tension(/gravity) stiffness
    
        # Element parameters
        self.R = R
        self.r = self.L/self.R
        self.element_mass = self.total_mass/self.R
        self.kx = 0
        self.ky = 0
        self.kr = self.total_kx/self.R
        self.cx = 40000/self.R
        self.element_inertia = 1/2*self.element_mass*(self.D_pipe/2)**2
        
        self.point_mass_mass = 50*10**6
        self.ky_m = self.total_kx*10000
        self.kx_m = self.total_kx*10

        self.origin = [0, 0]
        self.initial_conditions = np.zeros((self.R,2))
        self.initial_conditions[0,0] = 3/10 * np.pi/180
        #self.initial_conditions[0,1] = 3/10000 * np.pi/180


        ## solver attributes
        self.matrix_size = self.R*3+2
        self.t_start = 0
        self.t_end = 5

        self.eps_abs = 10**-6
        self.eps_rel = 10**-1

        self.g = sp.var("g")

        self.elements = dict()
        self.setup_elements()

        self.point_mass = dict()
        self.setup_point_mass()

        self.rotational_springs = [sp.var(f"k_r{i}") for i in range(1,self.R)]
        self.current_forcing = sp.Function("F_current")
        self.wave_forcing = sp.Function("F_waves")
        self.x0 = sp.Function("x_0")
        self.y0 = sp.Function("y_0")
        self.theta0 = sp.Function("theta_0")

        self.EOMS = []
        self.acceleration_matrix = []
        self.constrained_acceleration_matrix = []
        self.current_state = dict()

        self.setup_equations_of_motion()
        self.substitute_system_constraints()
        self.setup_current_state_dictionary()

    ## HELPERS
    def save_latex_eoms(self, fname="eoms_latex.txt"):
        eoms = self.sub_eoms_with_symbols(print_latex=True)
        eoms_str = "\n\n".join([sp.latex(eom, itex=True, mode="equation", mul_symbol="dot") for eom in eoms])
        eoms_str = eoms_str.replace("\\theta", "theta")
        eoms_str = eoms_str.replace("theta", "\\theta")

        with open(fname, "w") as f:
            f.write(eoms_str)


    def append_equation_to_latex(self, eq, fname="eoms_latex.txt"):
        with open(fname, "a") as f:
            f.writelines([sp.latex(eq, itex=True, mode="equation", mul_symbol="dot"), "\n\n\n"])


    def setup_elements(self):
        for i in range(1,self.R+1):
            self.elements[i] = dict()
            self.elements[i]["const"] = list(sp.var(f"m_{i} I_{i}"))
            self.elements[i]["vars"]  = [sp.Function(f"{j}_{i}") for j in ["x", "y", "theta"]]
            self.elements[i]["coeff"] = list(sp.var(f"k_x{i} k_y{i} c_{i} K_t{i}"))


    def setup_point_mass(self):
        # setup variables and degrees of freedom for the pointmass
        self.point_mass[0] = dict()
        M_var = sp.var("M")
        x_m, y_m = sp.Function("x_m"), sp.Function("y_m")
        self.point_mass[0]["const"] = [M_var]
        self.point_mass[0]["vars"] = [x_m, y_m]
        self.point_mass[0]["coeff"] = list(sp.var("k_xm k_ym"))


    ## EQUATION OF MOTION
    def get_element_lagrangian(self, K, P, i):
        x, y, theta = self.elements[i]["vars"]
        kx, ky, c, Kt = self.elements[i]["coeff"]
        L = K - P

        eom1 = sp.Eq(sp.simplify(L.diff(x(t).diff(t)).diff(t) - L.diff(x(t))), self.current_forcing(t))
        #eom1 = sp.Eq(sp.simplify(L.diff(x(t).diff(t)).diff(t) - L.diff(x(t))), 0)
        eom2 = sp.Eq(sp.simplify(L.diff(y(t).diff(t)).diff(t) - L.diff(y(t))), 0)
        eom3 = sp.Eq(sp.simplify(L.diff(theta(t).diff(t)).diff(t) - L.diff(theta(t))), 0)
    
        ## Substitute Kt = k + d/dt c
        # For now only damping in the x direction
        eom1 = eom1.subs(Kt*x(t), kx*x(t) + c*x(t).diff(t))

        return eom1, eom2, eom3


    def get_mass_lagrangian(self, K, P):
        x, y = self.point_mass[0]["vars"]
        kx, ky = self.point_mass[0]["coeff"]
        L = K - P
        self._kinetic = K
        self._potential = P
        self._lagragian = L

        #eom1 = sp.Eq(sp.simplify(L.diff(x(t).diff(t)).diff(t) - L.diff(x(t))), self.wave_forcing(t))
        eom1 = sp.Eq(sp.simplify(L.diff(x(t).diff(t)).diff(t) - L.diff(x(t))), 0)
        eom2 = sp.Eq(sp.simplify(L.diff(y(t).diff(t)).diff(t) - L.diff(y(t))), 0)

        # For now no mass damping

        return eom1, eom2


    def setup_kinetic_energy(self):
        M_var = self.point_mass[0]["const"][0]  
        x_m, y_m = self.point_mass[0]["vars"]
        K = sp.Rational(1,2) * M_var * sp.sqrt(x_m(t).diff(t)**2 + y_m(t).diff(t)**2)
        for i in self.elements.keys():
            x, y, theta = self.elements[i]["vars"]
            m, I = self.elements[i]["const"]
            K += sp.Rational(1,2) * m * sp.sqrt(x(t).diff(t)**2 + y(t).diff(t)**2) + sp.Rational(1,2) * I * theta(t).diff(t)**2
        
        self._kinetic_energy = K
        return self._kinetic_energy


    def setup_potential_energy(self):
        # Mass
        k_x, k_y = self.point_mass[0]["coeff"]
        x, y = self.point_mass[0]["vars"] 
        m = self.point_mass[0]["const"][0]
        P =  sp.Rational(1,2)*k_x*x(t)**2 + sp.Rational(1,2)*k_y*y(t)**2 + m*self.g*(y(t) - self.L)

        for i in self.elements.keys():
            x, y, theta = self.elements[i]["vars"]
            m, I = self.elements[i]["const"]
            kx, ky, c, Kt = self.elements[i]["coeff"]

            P +=  sp.Rational(1,2) * ky * (y(t) - self.r*((i-1)+sp.Rational(1,2)))**2 + sp.Rational(1,2) * Kt * x(t)**2 + m*self.g*(y(t) - self.r*((i-1)+sp.Rational(1,2)))
            if not i == 1:
                xm1, ym1, thetam1 = self.elements[i-1]["vars"]
                kr = self.rotational_springs[i-2]
                P += sp.Rational(1,2)*kr*(theta(t) - thetam1(t))**2

        self._potential_energy = P
        return self._potential_energy


    def setup_equations_of_motion(self):
        self.EOMS = []
        K = self.setup_kinetic_energy()
        P = self.setup_potential_energy()
        
        self.EOMS.extend(self.get_mass_lagrangian(K, P))
        for i in self.elements.keys():
            self.EOMS.extend(self.get_element_lagrangian(K, P, i))

        return self.EOMS

    
    ## GET ACCELERATION MATRIX
    def sub_eoms_with_symbols(self, print_latex=False):
        if not self.EOMS:
            print("Did not find the EOMs, setting them up")
            self.setup_equations_of_motion()
        
        new_EOMS = []
        self.all_accelerations = []
        self.all_velocities = []
        self.all_positions = []
        ## First substitute derivatives for symbols
        for j,eq in enumerate(self.EOMS):
            ## substitute mass derivatives
            xm, ym = self.point_mass[0]["vars"]
            xm_d, xm_dd, ym_d, ym_dd = sp.symbols("x_m_d x_m_dd y_m_d y_m_dd")
            if print_latex:
                xm_d, xm_dd, ym_d, ym_dd = sp.symbols("\\dot{x}_m \\ddot{x}_m \\dot{y}_m \\ddot{y}_m")
            if j == 0:
                self.all_accelerations.extend([xm_dd, ym_dd])
                self.all_velocities.extend([xm_d,ym_d])
                self.all_positions.extend([xm,ym])

            eq = eq.subs([(xm(t).diff((t,2)), xm_dd), (ym(t).diff((t,2)), ym_dd)])
            eq = eq.subs([(xm(t).diff(t), xm_d),  (ym(t).diff(t), ym_d)])

            xm_sym, ym_sym = sp.symbols("x_m y_m")
            eq = eq.subs([(xm(t), xm_sym), (ym(t), ym_sym)])

            self.point_mass[0]["symbols"] = {"x": [xm_sym, xm_d, xm_dd], "y":[ym_sym, ym_d, ym_dd]}

            for i,d in self.elements.items():
                self.elements[i]["symbols"] = dict()
                for v in d["vars"]:
                    vd, vdd = sp.symbols(f"{str(v)}_d {str(v)}_dd")
                    if print_latex:
                        v_main, v_sub = str(v).split("_")
                        vd, vdd = sp.symbols("\\dot{" + v_main +"}_" + v_sub + " \\ddot{" + v_main + "}_" + v_sub)

                    eq = eq.subs([(v(t).diff((t,2)), vdd), (v(t).diff(t), vd)])

                    v_sym = sp.symbols(str(v))
                    eq = eq.subs(v(t), v_sym)

                    symbol_key = str(v).split("_")[0]
                    self.elements[i]["symbols"][symbol_key] = [v_sym, vd, vdd]
                
                    if j == 0:
                        self.all_accelerations.append(vdd)
                        self.all_velocities.append(vd)
                        self.all_positions.append(v_sym)

            new_EOMS.append(eq)
        return new_EOMS


    def get_nonlin_acceleration_matrix(self):
        eoms = self.sub_eoms_with_symbols()
        mass_matrix = sp.matrices.zeros(self.matrix_size, self.matrix_size)
        stiffness_matrix = sp.matrices.zeros(self.matrix_size, self.matrix_size)
        force_matrix = sp.matrices.zeros(self.matrix_size, self.matrix_size)
        acceleration_matrix = sp.matrices.zeros(self.matrix_size, self.matrix_size)

        all_coeff = np.append(np.array([d["coeff"] for d in self.elements.values()]).flatten(), self.point_mass[0]["coeff"])
        all_coeff = np.append(all_coeff, self.rotational_springs)
        all_consts = np.append(np.array([d["const"] for d in self.elements.values()]).flatten(), self.point_mass[0]["const"])
        
        for m, eq in enumerate(eoms):
            for n  in range(self.matrix_size):
                ## MASS
                acc_symbol, vel_symbol, pos_symbol = self.all_accelerations[n], self.all_velocities[n], self.all_positions[n]
                available_acc = eq.free_symbols.intersection(self.all_accelerations)
                gravity_present = eq.free_symbols.intersection([self.g])

                if not acc_symbol in available_acc:
                    mass_matrix[m,n] = 0
                else:
                    # sub coupled accelerations with zero
                    zero_acc = available_acc.difference([acc_symbol])
                    temp_eq = eq.subs({(acc, 0) for acc in zero_acc})
                    
                    # sub spring and damping coeff with zero
                    temp_eq = temp_eq.subs({(coef,0) for coef in all_coeff})
                    #temp_eq = temp_eq.subs({(self.g, 0)})

                    # sub current acceleration symbol with 1
                    temp_eq = temp_eq.subs({(acc_symbol,1)})
                    if not temp_eq == True and not temp_eq == False:
                        mass_matrix[m,n] = temp_eq.lhs
                
                ## STIFFNESS
                available_pos = eq.free_symbols.intersection(self.all_positions)
                available_vel = eq.free_symbols.intersection(self.all_velocities)
                if not vel_symbol in available_vel and not pos_symbol in available_pos:
                    stiffness_matrix[m,n] = 0
                else:
                    # sub all coupled velocities with zero
                    zero_vel = available_vel.difference([vel_symbol])
                    temp_eq = eq.subs({(vel,0) for vel in zero_vel})
                    
                    # sub all coupled positions with zero
                    zero_pos = available_pos.difference([pos_symbol])
                    temp_eq = temp_eq.subs({(pos,0) for pos in zero_pos})
                    
                    # sub all mass and inertial constants with zero
                    temp_eq = temp_eq.subs({(const,0) for const in all_consts})
                    #temp_eq = temp_eq.subs({(self.g,0)})

                    if not temp_eq == True and not temp_eq == False:
                        stiffness_matrix[m,n] = temp_eq.lhs
                
                ## FORCING
                if eq.rhs:
                    ## TODO: Add gravity and forcing
                    print("Found forcing, not implemented yet!")
        
        self.mass_matrix = mass_matrix
        self.stiffness_matrix = stiffness_matrix
        self.force_matrix = force_matrix

        RHS = force_matrix - stiffness_matrix
        self.acceleration_matrix = mass_matrix.inv() * RHS

        return self.acceleration_matrix

    ## IMPLEMENT SYSTEM CONSTRAINTS
    def setup_system_constraints(self):
        all_constraints = []

        xm1 = self.x0
        ym1 = self.y0
        thetam1 = self.theta0

        #TODO: possibly save initial condition symbols
        x_sym_m1, x_d_m1, x_dd_m1 = sp.symbols("x0 x0_d x0_dd")
        y_sym_m1, y_d_m1, y_dd_m1 = sp.symbols("y0 y0_d y0_dd")
        theta_sym_m1, theta_d_m1, theta_dd_m1 = sp.symbols("theta0 theta0_d theta0_dd")
        
        for i,d in self.elements.items():
            x, y, theta = d["vars"]
            ## Get symbols
            x_sym, x_d, x_dd = d["symbols"]["x"]
            y_sym, y_d, y_dd = d["symbols"]["y"]
            theta_sym, theta_d, theta_dd = d["symbols"]["theta"]

            if i == 1:
                y_eq = ym1(t) + sp.Rational(1,2)*self.r*sp.cos(theta(t))
                x_eq = xm1(t) + sp.Rational(1,2)*self.r*sp.sin(theta(t))
            else:
                y_eq = ym1(t) + sp.Rational(1,2)*self.r*sp.cos(thetam1(t)) + sp.Rational(1,2)*self.r*sp.cos(theta(t))
                x_eq = xm1(t) + sp.Rational(1,2)*self.r*sp.sin(thetam1(t)) + sp.Rational(1,2)*self.r*sp.sin(theta(t)) 
            
            y_dot_eq = y_eq.diff(t)
            x_dot_eq = x_eq.diff(t)
        
            # Define constraints for x and y in terms of theta
            x_constraint = (x_sym, x_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (xm1(t).diff(t), x_d_m1), (xm1(t), x_sym_m1)]))
            y_constraint = (y_sym, y_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (ym1(t).diff(t), y_d_m1), (ym1(t), y_sym_m1)]))
            
            # Define constraints for x_dot and y_dot in terms of theta
            x_dot_constraint = (x_d, x_dot_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (x(t).diff(t), x_d_m1), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (xm1(t).diff(t), x_d_m1), (xm1(t), x_sym_m1)]))
            y_dot_constraint = (y_d, y_dot_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (y(t).diff(t), y_d_m1), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (ym1(t).diff(t), y_d_m1), (ym1(t), y_sym_m1)]))
            
            # Save constraints
            constraints =  [x_constraint, y_constraint, x_dot_constraint, y_dot_constraint ]
            self.elements[i]["constraints"] = constraints
            all_constraints.extend(constraints)

            # Save variables of previous x, y and theta, e.g.: x_(i-1)
            xm1 = x
            ym1 = y
            thetam1 = theta

            x_sym_m1, x_d_m1, x_dd_m1 = x_sym, x_d, x_dd
            y_sym_m1, y_d_m1, y_dd_m1 = y_sym, y_d, y_dd
            theta_sym_m1, theta_d_m1, theta_dd_m1 = theta_sym, theta_d, theta_dd

        ## Add constraints on the point mass
        x, y = self.point_mass[0]["vars"]
        x_sym, x_d, x_dd = self.point_mass[0]["symbols"]["x"]
        y_sym, y_d, y_dd = self.point_mass[0]["symbols"]["y"]

        y_eq = ym1(t) + sp.Rational(1,2)*self.r*sp.cos(thetam1(t))
        y_dot_eq = y_eq.diff(t)
        
        x_eq = xm1(t) + sp.Rational(1,2)*self.r*sp.sin(theta(t))
        x_dot_eq = x_eq.diff(t)

        x_constraint = (x_sym, x_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (xm1(t).diff(t), x_d_m1), (xm1(t), x_sym_m1)]))
        y_constraint = (y_sym, y_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (ym1(t).diff(t), y_d_m1), (ym1(t), y_sym_m1)]))
        
        # Define constraints for x_dot and y_dot in terms of theta
        x_dot_constraint = (x_d, x_dot_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (x(t).diff(t), x_d_m1), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (xm1(t).diff(t), x_d_m1), (xm1(t), x_sym_m1)]))
        y_dot_constraint = (y_d, y_dot_eq.subs([(theta(t).diff(t), theta_d), (theta(t), theta_sym), (y(t).diff(t), y_d_m1), (thetam1(t).diff(t), theta_d_m1), (thetam1(t), theta_sym_m1), (ym1(t).diff(t), y_d_m1), (ym1(t), y_sym_m1)]))

        constraints =  [x_constraint, y_constraint, x_dot_constraint, y_dot_constraint ]
        self.point_mass[0]["constraints"] = constraints
        all_constraints.extend(constraints)

        return all_constraints

    def substitute_system_constraints(self):
        if not self.acceleration_matrix:
            self.get_nonlin_acceleration_matrix()

        all_constraints = self.setup_system_constraints()
        self.constrained_acceleration_matrix = sp.matrices.zeros(self.matrix_size, self.matrix_size)
        
        for m in range(self.matrix_size):
            for n in range(self.matrix_size):
                eq = self.acceleration_matrix[m,n]
                while any(str(v).startswith("x_") or str(v).startswith("y_") for v in eq.free_symbols):
                    eq = eq.subs(all_constraints)

                self.constrained_acceleration_matrix[m,n] = eq
            
                for s in eq.free_symbols:
                    self.current_state[s] = None
        
        return self.constrained_acceleration_matrix
   

    ## SOLVE SYSTEM NUMERICALLY
    def setup_current_state_dictionary(self):
        for k, v in self.current_state.items():
            k_str = str(k)
            if k_str.startswith("m"):
                self.current_state[k] = self.element_mass
            elif k_str == "M":
                self.current_state[k] = self.point_mass_mass 
            elif k_str == "k_xm":
                self.current_state[k] = self.kx*100
            elif k_str.startswith("k_x"):
                self.current_state[k] = self.kx
            elif k_str.startswith("k_ym"):
                self.current_state[k] = self.ky_m
            elif k_str.startswith("k_y"):
                self.current_state[k] = self.ky
            elif k_str.startswith("k_r"):
                self.current_state[k] = self.kr
            elif k_str.startswith("c"):
                self.current_state[k] = self.cx
            elif k_str.startswith("I"):
                self.current_state[k] = self.element_inertia
            elif k_str == "g":
                self.current_state[k] = self.g_eff
            elif k_str == "x0":
                self.current_state[k] = self.origin[0]
            elif k_str == "y0":
                self.current_state[k] = self.origin[1]
            elif k_str == "x0_d":
                self.current_state[k] = self.origin[0]
            elif k_str == "y0_d":
                self.current_state[k] = self.origin[1]
            elif k_str.startswith("theta"): 
                j = 0
                str_split = k_str.split("_")
                if len(str_split) == 3:
                    j = 1
                i = int(str_split[1])-1
                self.current_state[k] = self.initial_conditions[i,j]
        return self.current_state


    def construct_q_from_theta(self):
        q = np.zeros((self.R*3+2, 2))
        for i,d in self.elements.items():
            j = i-1
            xsym, cx = d["constraints"][0]
            ysym, cy = d["constraints"][1]
            xdsym, cxd = d["constraints"][2]
            ydsym, cyd = d["constraints"][3]
            
            x = cx.evalf(subs=self.current_state)
            y = cy.evalf(subs=self.current_state)
            theta = self.current_state[d["symbols"]["theta"][0]]
            xd = cxd.evalf(subs=self.current_state)
            yd = cyd.evalf(subs=self.current_state)
            thetad = self.current_state[d["symbols"]["theta"][1]]

            self.current_state[xsym] = x
            self.current_state[ysym] = y
            self.current_state[xdsym] = xd
            self.current_state[ydsym] = yd

            q[2+3*j, 0] = x
            q[3+3*j, 0] = y 
            q[4+3*j, 0] = theta
            q[2+3*j, 1] = xd
            q[3+3*j, 1] = yd
            q[4+3*j, 1] = thetad
        
        cmxsym, cmx = self.point_mass[0]["constraints"][0]
        cmysym, cmy = self.point_mass[0]["constraints"][1]
        cmxdsym, cmxd = self.point_mass[0]["constraints"][2]
        cmydsym, cmyd = self.point_mass[0]["constraints"][3]

        q[0, 0] = cmx.evalf(subs=self.current_state)
        q[0, 1] = cmxd.evalf(subs=self.current_state)
        q[1, 0] = cmy.evalf(subs=self.current_state)
        q[1, 1] = cmyd.evalf(subs=self.current_state)
        return q


    def compute_accelerations(self, qn, t):
        #TODO: add time to current state
        for i,d in self.elements.items():
            j = i - 1
            theta_sym, theta_dot_sym = d["symbols"]["theta"][:2]
            self.current_state[theta_sym] = qn[4+3*j, 0]
            self.current_state[theta_dot_sym] = qn[4+3*j, 1]

        acc = []
        
        for m in range(self.matrix_size):
            eq = sum(self.constrained_acceleration_matrix[m,:])
            result = eq.evalf(subs=self.current_state)
            if type(result) == sp.core.numbers.NaN:
                print(self.all_accelerations[m], result)
                result = 0

            acc.append(result)
        return acc


    def qdot(self, t, qn):
        print(t)
        """
        We construct the qn matrix from the ground up, so starting at rod 1 up to rod j, ending with the point mass.
        qn = 
        [xm, xm_dot],
        [ym, ym_dot],
        [x1, x1_dot],
        [y1, y1_dot],
        [theta1, theta1_dot],
        ...
        [xj, xj_dot],
        [yj, yj_dot],
        [thetaj, thetaj_dot],
        """

        # calculate accelerations based on this dict and y_{n-1}
        qn = qn.reshape(self.R*3+2, 2)
        acc_vector = self.compute_accelerations(qn, t)

        # construct qdot
        qdot = np.array([[q[1], a] for q,a in zip(qn, acc_vector)])
        return qdot.flatten()


    def ivp_solver(self):
        # setup current state dictionary
        self.setup_current_state_dictionary()
        print("Starting state:")
        print(self.current_state)

        # setup q0 based on only theta0, x0 and y0
        q0 = self.construct_q_from_theta()
        print("q0:")
        print(q0)
        q0 = q0.flatten()

        # solve ivp
        start_time = datetime.datetime.now()
        sol = solve_ivp(self.qdot, (self.t_start, self.t_end), q0, atol=self.eps_abs, rtol=self.eps_rel)
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print("Calculation lasted: ", duration.total_seconds(), " seconds")
        self.rk45_sol = sol
        self.x_solution =           sol.y[[0, 4, 10, 16, 22, 28]]
        self.x_dot_solution =       sol.y[[1, 5, 11, 17, 23, 29]]
        self.y_solution =           sol.y[[2, 6, 12, 18, 24, 30]]
        self.y_dot_solution =       sol.y[[3, 7, 13, 19, 25, 31]]
        self.theta_solution =       sol.y[[   8, 14, 20, 26, 32]]
        self.theta_dot_solution =   sol.y[[   9, 15, 21, 27, 33]]

    ## Visualisation
    def plot_x_positions(self):
        for i, x in enumerate(self.x_solution):
            if i == 0:
                label = "m"
            else:
                label = i
            plt.plot(self.rk45_sol.t, x, label=label)
        plt.legend()
        plt.savefig("x_positions.png")
        plt.cla()

    def plot_y_positions(self):
        for i, y in enumerate(self.y_solution):
            if i == 0:
                label = "m"
            else:
                label = i
            plt.plot(self.rk45_sol.t, y, label=label)
        plt.legend()
        plt.title("Y positions")
        plt.grid(True, "both")
        plt.savefig("y_positions.png")
        plt.cla()

    def plot_theta_positions(self):
        for i, theta in enumerate(self.theta_solution):
            if i == 0:
                label = "m"
            else:
                label = i
            plt.plot(self.rk45_sol.t, theta, label=label)
        plt.legend()
        plt.savefig("theta_positions.png")
        plt.cla()

""" 
a1, a2 = sp.symbols("a1 a2")
eom1 = EoM1.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
eom2 = EoM2.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
MTRX = sp.linear_eq_to_matrix([eom1, eom2], [a1, a2])
M, F = MTRX[0], MTRX[1]

F_n = M.inv().dot(F)

t1, t2, t1dot, t2dot = sp.symbols("t1 t2 t1dot t2dot")
F_n1 = F_n[0].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})
F_n2 = F_n[1].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})





    def get_acceleration_function1(self):
        # solve for x_i_dd, y_i_dd, theta_i_dd
        #eoms, dd_symbols = self.sub_eoms_with_symbols()
        #print(dd_symbols)
        #sol = sp.nonlinsolve(eoms, dd_symbols)
        #print(sol)
        dd_symbols = []
        new_eoms = []
        for j,eq in enumerate(self.EOMS):

            xm, ym = self.point_mass[0]["vars"]
            xm_dd, ym_dd = sp.symbols("x_m_dd y_m_dd")
            if j == 0:
                dd_symbols.extend([xm_dd, ym_dd])

            eq = eq.subs([(xm(t).diff((t,2)), xm_dd), (ym(t).diff((t,2)), ym_dd)])
            for i,d in self.elements.items():
                for v in d["vars"]:
                    vdd = sp.symbols(f"{str(v)}_dd")
                    eq = eq.subs([(v(t).diff((t,2)), vdd)])
                    if j == 0:
                        dd_symbols.append(vdd)
        
            new_eoms.append(eq)

        sol = sp.nonlinsolve(new_eoms, dd_symbols)
        return sol

    def solve_nonlin_system(self):
        if not self.EOMS:
            print("Setting up EOMS")
            self.setup_equations_of_motion()
        
        print("Substituting symbols")
        self.nonlinear_eoms = self.sub_eoms_with_symbols()
        self.nonlinear_eoms += self.add_initial_conditions()

        all_symbols = self.point_mass[0]["symbols"]
        for d in self.elements.values():
            all_symbols.extend(d["symbols"])

        ### NONLIN SOLVER
        #print(f"Found {len(self.nonlinear_eoms)} equations and {len(all_symbols)} variables")
        #print("Getting solution")
        #sol = sp.nonlinsolve(self.nonlinear_eoms, all_symbols)
        #return sol
        print("NOT IMPLEMENTED")
        return False

    def solve_ode_system(self):
        sol_vars = np.array(self.point_mass[0]["vars"])
        sol_vars = np.append(sol_vars, [ x["vars"] for x in self.elements.values() ]) 
        sol_vars = [x(t) for x in sol_vars]
        sol = sp.solvers.ode.dsolve(self.EOMS, func=sol_vars)
        return sol

    def solve_ode(self, i):
        sol_vars = np.array(self.point_mass[0]["vars"])
        sol_vars = np.append(sol_vars, [ x["vars"] for x in self.elements.values() ]) 
        sol_vars = [x(t) for x in sol_vars]
        eq = self.EOMS[i]
        sol_var = sol_vars[i]
        print(sol_var, "\t\t", eq, "\n\n")

        sol = sp.solvers.ode.dsolve(self.EOMS[i], func=sol_vars[i])
        return sol
        
    def add_initial_conditions(self):
        pass

    def add_boundary_conditions(self):
        pass

"""
