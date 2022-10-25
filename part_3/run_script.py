# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:59:44 2022

@author: thijs
"""
from FEM import ContinousBeam
from matplotlib import pyplot as plt
beam = ContinousBeam()
print(beam.A)
print(beam.I)
print(f"{beam.EI_pipe:.2e}")
print(f"{beam.EA_pipe:.2e}")

beam.get_solution_FEM()
plt.plot(beam.solution.t/3600, beam.solution.y[2])
