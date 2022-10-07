# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:56:58 2022

@author: thijs
"""
import numpy as np
import sympy as sp

sp.var("m r g t c x0")
theta1 = sp.Function("theta1")(t)
theta2 = sp.Function("theta2")(t)
x1 = x0 + r*sp.cos(theta1)
x2 = x1 + r*sp.cos(theta2)
y1 = r*sp.sin(theta1)
y2 = y1+r*sp.sin(theta2)

K = 1/2*m*(sp.diff(x1, t)**2+sp.diff(y1, t)**2 + sp.diff(x2, t)**2 + sp.diff(y2, t)** 2)
P = m*g*(y1+y2)+ 1/2*c*r*(sp.diff(x1, t))**2

L = K-P
EoM1 = sp.simplify(sp.diff(sp.diff(L, sp.diff(theta1, t)), t)-sp.diff(L, theta1))
EoM2 = sp.simplify(sp.diff(sp.diff(L, sp.diff(theta2, t)), t)-sp.diff(L, theta2))


system = [EoM1,EoM2]
solve = sp.solve(system, [sp.diff(theta1, (t, 2)), sp.diff(theta2, (t, 2))], dict = True)
EoM1 = solve[0][sp.diff(theta1, (t, 2))]
EoM2 = solve[0][sp.diff(theta2, (t, 2))]


theta1sym, theta2sym, tsym, theta1dot, theta2dot = sp.symbols("ts1 ts2 tsym t1dot t2dot")
EoM1  = EoM1.evalf(subs = {t : tsym, theta1: theta1sym, theta2: theta2sym, sp.diff(theta1, t): theta1dot,sp.diff(theta2, t): theta2dot  , m :0.8, g : 9.81, r: 1, c : 0.8})
EoM2  = EoM2.evalf(subs = {t : tsym, theta1: theta1sym, theta2: theta2sym, sp.diff(theta1, t): theta1dot,sp.diff(theta2, t): theta2dot  , m :0.8, g : 9.81, r: 1, c : 0.8})
print(EoM1)

def qdot(t,q):
    q_new = np.zeros(4)
    q_new[0] = q[2]
    q_new[1] = q[3]
    q_new[2] = EoM1.evalf(subs={theta1dot: q[2], theta2dot: q[3], theta1sym: q[0], theta2sym: q[2], tsym: t})
    q_new[3] = EoM2.evalf(subs={theta1dot: q[2], theta2dot: q[3], theta1sym: q[0], theta2sym: q[1], tsym: t})
    
    return q_new



def FE(fun, tspan, y0):
    q = np.zeros((len(tspan), 4))
    q[0] = y0
    for idx, t in enumerate(tspan[:-1]):
        q_dot_i = fun(t, q[idx])
        q[idx+1] = q[idx] + dt*q_dot_i
    return q

dt = 0.5
tspan = np.arange(0, 5, dt)

FE_sol = FE(qdot,tspan, [0.5, 0, 0, 0])


plt.plot(sol.t,r*np.cos(sol.y[0]))
plt.plot(sol.t,r*np.cos(sol.y[1]))
# plt.plot(sol.t,sol.y[2])
# plt.plot(sol.t,sol.y[3])

# from scipy.integrate import solve_ivp
# sol = solve_ivp(fun=qdot,t_span=[0, 5],y0=[0.5, 0, 0, 0])




# plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1])
# plt.plot(sol.t,sol.y[2])
# plt.plot(sol.t,sol.y[3])
