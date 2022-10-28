# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:09:58 2022

@author: thijs
"""
import numpy as np
from FEM import ContinousBeam
from FEM import Cable
from matplotlib import pyplot as plt

#%%
# beam = ContinousBeam()
# beam.get_solution_FEM()
# #plt.plot(beam.solution.t/3600, beam.displacements[50])
# #plt.twinx()
# plt.plot(beam.solution.t/3600, beam.velocities[50])
# plt.twinx()
# plt.plot(beam.solution.t/3600, beam.accelerations[50], color= "red")

# %%

fig, ax = plt.subplots(2, 3, figsize = (30, 15))
karr = [0,50,90]
sagarr = [10, 10, 10]
print(ax.shape)
a = [0,0,0]
for i in range(3):
    ml = Cable()
    ml.ksoil = karr[i]
    ml.SAG = sagarr[i]
    ml.setup_system()
    ml.static_solver()
    print(ml.ksoil)
    print(np.min(ml.TENSION))
    try:
        ax[0, i].plot(ml.sol[0], ml.sol[1], color = 'black')
        
    except AttributeError:
        print("No attribute sol")
    ax[0, i].hlines(0, -ml.R*1.1, ml.R*1.1, color= 'brown')
    ax[0, i].set_title(f"Solution after {ml.kIter} iterations and ksoil = {karr[i]} N/m")
    ax[0, i].set_xlabel("x [m]")
    
    ax[0, i].plot([-ml.R, 0, ml.R], [-ml.anchor_depth, ml.H, -ml.anchor_depth], 'vr')
    ax[0, i].set_ylabel("y [m]")
    ax[1, i].plot(ml.TENSION)
    ax[1, i].set_title("Tension")
    ax[1, i].set_xlabel("x [m]")
    ax[1, i].set_ylabel("T [N]")

plt.savefig('Soil Model.jpg')
