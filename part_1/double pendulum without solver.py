# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:52:00 2022

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

a1, a2 = sp.symbols("a1 a2")
eom1 = EoM1.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
eom2 = EoM2.evalf(subs={sp.diff(theta1, (t, 2)): a1, sp.diff(theta2, (t, 2)): a2})
MTRX = sp.linear_eq_to_matrix([eom1, eom2], [a1, a2])
M, F = MTRX[0], MTRX[1]

F_n = M.inv().dot(F)

t1, t2, t1dot, t2dot = sp.symbols("t1 t2 t1dot t2dot")
F_n1 = F_n[0].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})
F_n2 = F_n[1].evalf(subs = {theta1: t1, theta2: t2, sp.diff(theta1, t): t1dot, sp.diff(theta2, t):t2dot, c : 0.1, m:0.5, r:1, g : 9.81})


print(sp.python(F_n1))

# def qdot(t,q):
#     q_dot = np.zeros(4)
    
#     q_dot[0] = q[2]
#     q_dot[1] = q[3]
#     q_dot[2] = F_n.evalf(subs = {theta1: q[0], theta2: q[1], t1dot: q[2], t2dot: q[3] })
#     q_dot[3] = F_n.evalf(subs = {theta1: q[0], theta2: q[1], t1dot: q[2], t2dot: q[3]})
#     return q_dot


# from scipy.integrate import solve_ivp
# sol = solve_ivp(fun=qdot,t_span=[0, 5],y0=[0.5, 0, 0, 0])

# plt.plot(sol.t,sol.y[0])
# plt.plot(sol.t,sol.y[1])
# plt.plot(sol.t,sol.y[2])
# plt.plot(sol.t,sol.y[3])

