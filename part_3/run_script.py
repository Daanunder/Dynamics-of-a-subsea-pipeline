# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:59:44 2022

@author: thijs
"""
from FEM import ContinousBeam

beam = ContinousBeam()
beam.get_forcing()
sol = beam.get_solution_FEM()
print(sol.shape)