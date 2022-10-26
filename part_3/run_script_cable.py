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
ml.plot_cable()
ml.static_solver()

ml.plot_tension()
ml.g = 5000
ml.define_external_forcing()
ml.static_solver()
ml.plot_tension()
