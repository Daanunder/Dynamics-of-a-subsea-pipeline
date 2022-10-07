# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 17:51:55 2022

@author: thijs
"""
import numpy as np
import sympy as sp

sp.var("m c k t")
u = sp.Function("u")(t)
EoM = m*sp.diff(u, t, 2)+c*sp.diff(u, t)+k*u 




EOM = sp.solve(EoM, sp.diff(u, (t, 2)))


usym, tsym, vsym = sp.symbols("usym tsym vsym")
EOM_new  = EOM[0].evalf(subs = {t : tsym, u: usym, sp.diff(u, t): vsym , m :1.5, c:0.05, k:0.8})
             
def qdot(t,q):
    vt = q[1]
    at = EOM_new.evalf(subs={vsym: q[1], usym: q[0], tsym: t})
    return [vt,at]

from scipy.integrate import solve_ivp
sol = solve_ivp(fun=qdot,t_span=[0,40],y0=[4,0], t_eval = np.linspace(0, 40, 100))

import matplotlib.pyplot as plt
plt.plot(sol.t,sol.y[0])

