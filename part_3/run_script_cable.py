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

ml = Cable()
ml.static_solver()
ml.plot_cable()

x = np.linspace(0, 15, 100)
z = -5.128063249*np.cosh(0.1950053951*x - 2.345617545) + 7.013369948

plt.plot(x, -z,linewidth = 3, linestyle='dashed', color = 'mediumturquoise' ,label = 'Analytical solution')
plt.plot(-x, -z,linewidth = 3,linestyle='dashed',color = 'mediumturquoise')
plt.legend()

plt.savefig("analytical solution.jpg")