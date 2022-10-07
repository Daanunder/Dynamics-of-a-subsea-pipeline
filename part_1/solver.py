# Necessary packages to solve the ODE's numerically
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
import sympy as sp
import sympy.physics.mechanics as mech
from sympy.utilities.lambdify import lambdify
from sympy.abc import t

mech.init_vprinting()


class SDOF_system(object):

    """System class describing SDOF system parameters, equation of motion and solvers"""
    
    ## FORCING
    # current forcing, force directly imposed on the mass
    def current1(self, t=None, symbolic=False):
        if symbolic:
            t, A, w = sp.var("t A w")
            return A*sp.cos(w*t), self.a_current1, self.omega_f1
        return self.a_current1*np.sin(self.omega_f1*t)

    # current forcing, force directly imposed on the mass
    def current2(self, t=None, symbolic=False):
        if symbolic:
            t, A, w = sp.var("t A w")
            return A*sp.sin(w*t), self.a_current2, self.omega_f2
        return self.a_current2*np.sin(omega_f2*t)

    # surface waves forcing, forced displacement of the left boundary
    def waves1(self, t=None, symbolic=False):
        if symbolic:
            t, A, w, k1, c1 = sp.var("t A w k1 c1")
            return A*sp.cos(w*t)*k1 - A*2*sp.sin(w*t)*c1, self.a_waves, self.omega_f3
        return self.a_waves*np.cos(omega_f3*t)*self.k1 - self.a_waves*2*np.sin(omega_f3*t)*self.c1

    def __init__(self):
        self.rho_water = 1000

        # Pipe material parameters
        self.t = 0.004 # m
        self.L = 5000 #m  
        self.D = 0.5 #m
        self.rho_steel = 7850

        #area properties
        self.m =  np.pi * self.D * self.t * self.L * self.rho_steel             # [kg]
        self.I = 1/2*self.m/self.L*self.D**2
        self.E = 210000
        self.EI = self.E*self.I

        # ODE parameters
        self.g_eff = (self.rho_steel - self.rho_water)/self.rho_steel * 9.81
        self.k1 = 16*self.EI/(self.L/2)**2 + 2*self.m*self.g_eff/self.L   # [N/m] Structural Bending stifness + tension(/gravity) stiffness
        self.k2 = self.k1                                  # mass is lumped in the middle so both sides are just as stiff
        self.c1 = 1000                                # [Ns/m] guess of damping
        self.c2 = self.c1                                  # Also both damping are equal

        self.k = self.k1+self.k2
        self.c = self.c1+self.c2

        # force parameters (necessary for analytical solution)
        self.a_current1 = 10000
        self.a_current2 = 10000
        self.a_waves = 0
        self.omega_f1 = 1/200
        self.omega_f2 = 1/400
        self.omega_f3 = 1/2

        # parameters for analytical solution
        self.omega_n = np.sqrt((self.k1+self.k2)/self.m)
        self.zeta = (self.c1+self.c2)/(2*self.m*self.omega_n)
        self.omega_1 = self.omega_n*np.emath.sqrt(1-self.zeta**2)

        # Time interval
        self.t_0 =       0       
        self.t_f =       2000    # final time [s]
        self.dt =        0.001    # time step size [s]

        self.tspan =     np.arange(self.t_0, self.t_f, self.dt)   # vector of evaluation points [s]

        # Initial conditions
        self.init_disp = 10
        self.init_velo = 0
        self.q_0 = np.array([self.init_disp, self.init_velo])

        self.event_dict = {
                # save events in the form: 'name':[start, stop, function]
                "current1":[0, 250, self.current1, self.omega_f1],
                "current2":[250, self.t_f, self.current2, self.omega_f2],
                "waves1":[0,0, self.waves1, 0]
                }
    
    def get_force_at_time_t(self, t):
        f = 0
        for (start, stop, forcing) in self.event_dict.values():
            if (t >= start and t < stop):
                f += forcing(t)
        return f

    # Matrix to use the displacement and velocity to obtain the acceleration
    def qdot(self, t, q):
        q_dot =  np.array([[0,1],[-self.k/self.m, -self.c/self.m]])@q + np.array([0, self.get_forcing(t)/self.m])
        return q_dot
    

    def get_analytical_solution(self, q0=None, forcing_func=None):
        if not q0:
            q0 = self.q_0
        if forcing_func == None:
            ft, A_ft, w_ft = self.current1(symbolic=True)
        else:
            ft, A_ft, w_ft = forcing_func()
        
        xf, A_xf, w_xf = self.waves1(symbolic=True)

        sp.var("t m k1 k2 c1 c2")
        x = sp.Function("x")(t)

        K = sp.Rational(1,2)*m*x.diff(t)**2
        P = sp.Rational(1,2)*k2*x**2 + sp.Rational(1,2)*k1*(x-xf)**2 
        L = K - P
        Q = c1*(x.diff(t) - xf.diff(t)) + c2*x.diff(t)
        eom = L.diff(x.diff(t)).diff(t) - L.diff(x) - ft - Q

        ## Solve using dsolve
        xt = sp.Function("xt")
        eom = eom.evalf(subs={x: xt(t)})
        sol = sp.dsolve(eom, xt(t), ics={xt(0): q0[0], xt(t).diff(t).subs(t, 0): q0[1]})
        
        pos_out = 0
        velos_out = 0
        #pos_func = lambdify(t, sol.rhs, 'numpy')
        #pos_out = pos_func(self.tspan)

        return sol, pos_out, velos_out

#a1, a2 = sp.symbols("a1 a2")
#eom1 = EoM1.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
#eom2 = EoM2.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
#MTRX = sp.linear_eq_to_matrix([eom1, eom2], [a1, a2])
#M, F = MTRX[0], MTRX[1]
#
#F_n = M.inv().dot(F)
#
#t1, t2, t1dot, t2dot = sp.symbols("t1 t2 t1dot t2dot")
#F_n1 = F_n[0].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})
#F_n2 = F_n[1].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})


    def get_an_event_solution(event_array, y0):
        x, v = [], []

        for (start, stop, omega_f) in event_array:
            tspan = np.arange(start,stop,dt)
            x_next, v_next = get_an_solution(tspan, y0, omega_f)
            x.extend(x_next)
            v.extend(v_next)
            y0 = [x[-1], v[-1]]

        return np.array(x), np.array(v)


class solver(SDOF_system):
    def FE(self, fun, tspan, y0):
        q = np.zeros((len(tspan), 2))
        q[0] = y0
        for idx, t in enumerate(tspan[:-1]):
            q_dot_i = fun(t, q[idx])
            q[idx+1] = q[idx] + dt*q_dot_i
        return q

    def q_ddot_approx(q_dot_n, q_dot_nm1, dt):
        q_ddot_n = (q_dot_n - q_dot_nm1)/dt
        return q_ddot_n


    def FE_variable_t(fun, q_0, t_0, t_f, dt_0=100, eps_abs=0.01, eps_rel=0.1, var_factor=1):
        timestep_configured = False
        # Find appropriate timestep size to start with
        i = 0
        while not timestep_configured:
            # 1. Compute the first iteration of the numerical scheme with an arbitrarily small timestep
            q_dot_0 = fun(t_0, q_0)
            q_1 = q_0 + dt_0*q_dot_0

            # 2. Compute the proportional treshold $\tau = \epsilon_{rel}|q|$ and check if it is smaller than the absolute maximum $\epsilon_{abs}$.
            tau_rel = eps_rel * min([abs(i) for i in q_1])

            # 3 If not, $\Delta t_new = \frac{1}{2}\Delta t_old$ and start at step 1, else continue to step 4. 
            if tau_rel < eps_abs:
                timestep_configured = True
                print("Time step configured to: ", dt_0, "s")
            else:
                if i == 0:
                    print("WARNING - Initial step size badly configured, adjusting the size to conform to the given tresholds eps_abs and eps_rel")
                    i=1
                dt_0 = 0.5*dt_0

        # 4*. Compute an approximation for $\ddot{q}$ based on the computed state variable $\dot{q}$.
        # Note that this approximation based on $\dot{q}_1$ is only meant for $\ddot{q}_0$
        q_dot_1 = fun(dt_0, q_1)
        q_dot_nm1 = q_dot_1
        dt_n = dt_0

        q = []
        t = []
        t_n = t_0
        q_n = q_0
        while t_n < t_f:
            # Store time
            t.append(t_n)
            q.append(q_n)

            q_dot_n = fun(t_n, q_n)

            # 4. Compute an approximation for $\ddot{q}$ based on the computed state variable $\dot{q}$.
            q_ddot_n =  q_ddot_approx(q_dot_n, q_dot_nm1, dt_n)

            # 5. Compute the time step size based on $\ddot{q}$ and run the numerical scheme.
            tau = max(eps_rel*min([abs(i) for i in q_n]), eps_abs)
            dt_tau = np.sqrt(2*tau/min([abs(i) for i in q_ddot_n]))
            dt_n = (dt_n + dt_tau*var_factor) / (var_factor + 1)

            # 6. Compute value of $q_{n+1}$
            q_n = q_n + dt_n * q_dot_n

            # Save $\ddot{q}_{n-1}$ for next iteration
            q_dot_nm1 = q_dot_n


            t_n += dt_n

        return np.array(q),np.array(t)











"""

sol = solve_ivp(qdot,(t_0, t_f),q_0, t_eval=tspan, method='RK45', dense_output=True, rtol=0.00001, atol=0.000000001)

FE_sol = FE(qdot,tspan,q_0)

#an_sol = get_an_solution(tspan, q_0, omega_f1)
an_sol = get_an_event_solution(event_array, q_0)





fig, axes = plt.subplots(2,3, figsize=(18,12))
fig.tight_layout(pad=7.0)
ax1, ax2, ax3 = axes[0]
ax4, ax5, ax6 = axes[1]

# Runga Kutta
ax1.plot(sol.t,sol.y[0], label='displacement')
ax1.set_ylabel('Displacement [m]')
ax12 = ax1.twinx()
ax12.plot(sol.t,sol.y[1], 'tab:orange', label='velocity')
ax12.set_ylabel('Velocity [m/s]')
ax1.set_xlabel('Time [s]')
ax1.grid()
ax1.set_title('Runga-Kutta method')
ax1.legend()
ax12.legend(bbox_to_anchor=(1, 0.95))

# Forward Euler
ax2.plot(tspan,FE_sol[:, 0], label='displacement')
ax2.set_ylabel('Displacement [m]')
ax21 = ax2.twinx()
ax21.plot(tspan, FE_sol[:, 1], 'tab:orange', label='velocity')
ax21.set_ylabel('Velocity [m/s]')
ax2.set_xlabel('Time [s]');
ax2.grid()
ax2.set_title('Forward Euler method')
ax2.legend()
ax21.legend(bbox_to_anchor=(1, 0.95))


# Analytical
ax3.plot(tspan, an_sol[0], label='displacement')
ax3.set_ylabel('Displacement [m]')
ax31 = ax3.twinx()
ax31.plot(tspan, an_sol[1], 'tab:orange', label='velocity')
ax31.set_ylabel('Velocity [m/s]')
ax3.set_xlabel('Time [s]');
ax3.grid()
ax3.set_title('Analytical')
ax3.legend()
ax31.legend(bbox_to_anchor=(1, 0.95))


# RK45 error
ax4.plot(tspan, abs(an_sol[0]-sol.y[0]), label='displacement')
ax4.set_ylabel('Displacement [m]')
ax41 = ax4.twinx()
ax41.plot(tspan, abs(an_sol[1]-sol.y[1]), 'orange', label='velocity')
ax41.set_ylabel('Velocity [m/s]')
ax4.set_xlabel('Time [s]');
ax4.set_title('Error RK45')
ax4.legend()
ax41.legend(bbox_to_anchor=(1, 0.95))


# FE error
ax5.plot(tspan,abs(an_sol[0]-FE_sol[:,0]), label='displacement')
ax5.set_ylabel('Displacement [m]')
ax51 = ax5.twinx()
ax51.set_ylabel('Velocity [m/s]')
ax51.plot(tspan,abs(an_sol[1]-FE_sol[:,1]), 'orange', label='velocity')
ax5.set_xlabel('Time [s]');
ax5.set_title('Error FE')
ax5.legend()
ax51.legend(bbox_to_anchor=(1, 0.95))


# Difference FE & RK45
ax6.plot(tspan,abs(FE_sol[:,0]-sol.y[0]), label='displacement')
ax6.set_ylabel('Displacement [m]')
ax61 = ax6.twinx()
ax61.plot(tspan,abs(FE_sol[:,1]-sol.y[1]), 'orange', label='velocity')
ax61.set_ylabel('Velocity [m/s]')
ax6.set_xlabel('Time [s]');
ax6.set_title('Difference FE & RK45')
ax6.legend()
ax61.legend(bbox_to_anchor=(1, 0.95))


plt.show()





sol2 = solve_ivp(qdot,(t_0, t_f),q_0, method='RK45', dense_output=True,  rtol=(0.001, 0.01), atol=(0.001, 0.001))
timesteps = np.round(np.diff(sol2.t),9)

fig,(ax1,ax2) =plt.subplots(1,2, figsize=(16,6))
ax1.plot(sol2.t[1:], timesteps)
ax1.set_title("Variable time step size used by the RK-45 method")
ax1.set_xlabel('Evaluation time [s]')
ax1.set_ylabel('Time step size [s]')
ax1.grid()


index = [max(int(i-1),0) for i in sol2.t/dt]
sol_slice_displacement = an_sol[0][index]
sol_slice_velocity = an_sol[1][index]

ax2.plot(sol2.t,abs(sol_slice_displacement - sol2.y[0]), label='displacement')
ax2.set_ylabel('Displacement [m]')
ax21 = ax2.twinx()
ax21.plot(sol2.t,abs(sol_slice_velocity - sol2.y[1]), 'orange', label='velocity')
ax21.set_ylabel('Velocity [m/s]')
ax2.set_xlabel('Time [s]');
ax2.set_title('Error when variable time stepping is used')
ax2.legend()
ax2.grid()
ax21.legend(bbox_to_anchor=(1, 0.95))






FE_var_t_sol, FE_var_t = FE_variable_t(qdot, q_0, t_0, t_f, eps_rel=0.000001, eps_abs=0.00000001, var_factor=.1)

timesteps = np.round(np.diff(FE_var_t),9)

fig, [[ax3, ax1],[ax4,ax2]] = plt.subplots(2,2, figsize=(16,12))
fig.tight_layout(pad=7.0)

ax3.plot(FE_var_t, FE_var_t_sol[:,0], label='displacement')
ax3.set_title("Solution with variable time step and FE scheme")
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Displacement [m]')
ax31 = ax3.twinx()
ax31.plot(FE_var_t,FE_var_t_sol[:,1], 'orange', label='velocity')
ax31.set_ylabel('Velocity [m/s]')
ax3.grid()

ax1.plot(FE_var_t[1:], timesteps)
ax1.set_title("Variable time step size used by the FE method")
ax1.set_xlabel('Evaluation time [s]')
ax1.set_ylabel('Time step size [s]')
ax1.grid()

an_dt = 0.01
index = [min(max(int(i-1),0), int(t_f/an_dt-1)) for i in FE_var_t/an_dt]

an_sol_slice_displacement = an_sol[0][index]
an_sol_slice_velocity = an_sol[1][index]

FE_sol_slice_displacement = FE_sol[index,0]
FE_sol_slice_velocity = FE_sol[index,1]

ax2.plot(FE_var_t,abs(an_sol_slice_displacement - FE_var_t_sol[:,0]), label='displacement')
ax2.set_ylabel('Displacement [m]')
ax21 = ax2.twinx()
ax21.plot(FE_var_t,abs(an_sol_slice_velocity - FE_var_t_sol[:,1]), 'orange', label='velocity')
ax21.set_ylabel('Velocity [m/s]')
ax2.set_xlabel('Time [s]');
ax2.set_title('Error when variable time stepping is used')
ax2.legend()
ax2.grid()
ax21.legend(bbox_to_anchor=(1, 0.95))

ax4.plot(FE_var_t,abs(FE_sol_slice_displacement - FE_var_t_sol[:,0]), label='displacement')
ax4.set_ylabel('Displacement [m]')
ax41 = ax4.twinx()
ax41.plot(FE_var_t,abs(FE_sol_slice_velocity - FE_var_t_sol[:,1]), 'orange', label='velocity')
ax41.set_ylabel('Velocity [m/s]')
ax4.set_xlabel('Time [s]');
ax4.set_title('Difference with regular FE method')
ax4.legend()
ax4.grid()
ax41.legend(bbox_to_anchor=(1, 0.95))









f_min, f_max = 0.05, 0.2
resolution = 100
f = np.linspace(f_min, f_max, resolution)

Hm0 = 4.
Tp = 15.
E_js = jonswap(f, Hm0, Tp)
plt.plot(f, E_js)

num_of_comp = 100
df = abs(f_min - f_max)/num_of_comp

sample_index = np.random.randint(0, 100, size=num_of_comp)
sample_freq = f[sample_index]
sample_amplitude = 4*np.sqrt(df*E_js[sample_index]) # 4 sqrt(m0)
phases = np.random.uniform(0, np.pi*2, num_of_comp)

forcing = np.sum([ sample_amplitude[i] * np.cos(sample_freq[i]*tspan + phases[i]) for i in range(len(sample_index))], axis=0)

plt.plot(tspan, forcing)

"""

