# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp

n = 5

sp.var("t m r g ")

x_ = np.zeros(n, dtype = object)
y_ = np.zeros(n, dtype = object)
phi_ = np.zeros(n, dtype = object)

vsym_ = np.zeros(n, dtype = object)
phisym_ = np.zeros(n, dtype = object)
for i in range(n):
    phi_[i] = sp.Function(f"phi{i+1}")(t)
    
    # later for simulation used
    vsym_[i], phisym_[i] =sp.symbols(f"vsym{i} phisym{i}")

x_[0] = r/2*sp.cos(phi_[0])
for i in range(1, n):
    x_[i] = x_[i-1] + r/2*sp.cos(phi_[i-1]) + r/2 *sp.cos(phi_[i])
    y_[i] = y_[i-1] + r/2*sp.sin(phi_[i-1]) + r/2 *sp.sin(phi_[i])


K = 0
P = 0
for i in range(n):
    K += 1/2*(m) *(sp.diff(x_[i], t)**2+sp.diff(y_[i], t))
    P +=(m) * g * y_[i]

L = K-P
EoM = np.zeros(n, dtype=object)
for i, phi in enumerate(phi_):
    EoM[i] = sp.diff(sp.diff(L, sp.diff(phi, t)), t)- sp.diff(L, phi)
    EoM[i] = sp.solve(EoM[i], sp.diff(phi, (t, 2)))[0]


print(EoM[0])

tsym = sp.symbols("tsym")
for idx, e in enumerate(EoM):
    for i, vsym in enumerate(vsym_):
        EoM[idx] = EoM[idx].evalf(subs={sp.diff(phi_[i], t)  : vsym, phi_[i]: phisym_[i] })
    #EoM[idx] = e.evalf(subs={t: tsym,  g: 9.81, r: 1.0, m:1.0})

print(EoM[0])

# =============================================================================
# def qdot(t, q):
#     at = np.empty(n, dtype=object)
#     vt =q[n:] 
#     for i in range(len(EoM)):
#         at[i] = EoM[i].evalf(subs={tsym : t, phi_[1]: q[1], phi_[2]: q[2], phi_[3]: q[3], phi_[4]: q[4], phi_[5]:q[5]: })
#         print(at[i])
#     return np.concatenate(vt, at)
# 
# 
# 
# from scipy.integrate import solve_ivp
# sol = solve_ivp(fun=qdot,t_span=[0,10],y0=[1,0, 0, 0, 0,0,0,0,0,0])
# 
# import matplotlib.pyplot as plt
# plt.plot(sol.t,sol.y[0])
# =============================================================================
