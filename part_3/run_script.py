# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:59:44 2022

@author: thijs
"""
import numpy as np
from FEM import ContinousBeam
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size':18})

# %%
beam = ContinousBeam()

fig, ax = plt.subplots(2, 3, figsize= (20,8))
fig.suptitle('Response of first 3 modes to static excitations according to those modes')

for i in range(3):
    beam.mode_force = i+1
    beam.setup_system()
    beam.get_solution_FEM()
    for p in range(3):
        ax[0, i].plot(beam.solution.t/3600, beam.solution.y[p], label = f"Mode {p+1}")
        ax[0, i].legend(loc='upper right')
        x = np.arange(0, 5100, 100)
        y = 0.05* np.sin(x*np.pi*(i+1)/(5000))
        ax[1, i].plot(x, y, color = 'red')
        ax[1, i].set_ylim(-0.06, 0.06)
        ax[1, i].hlines(0, 0, 5000, color= 'black')
    ax[0, i].set_xlabel('t (hours)')
    ax[0, i].set_ylabel('Γ')
    ax[1, i].set_xlabel('x (m)')
    ax[1, i].set_ylabel('q (N/m)')
plt.tight_layout()

plt.savefig('Static Modal Excitation.jpg')

 