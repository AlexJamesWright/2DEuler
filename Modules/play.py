"""
play.py

script to test, play, and explore the functionalily of simulation.py
"""


from simulation import simulation
from initialFunctions import initialFunc
from cells import cells
from model import TwoFluidEMHD
from timeEv import backEulerRK2, SSP2
from boundaryConditions import periodic, outflow
from fluxApprox import fluxSplitting
from sourceTerms import sources, twoFluidDivClean
import numpy as np
import time


if __name__ == '__main__':
    timeStart = time.time()

    # Set up problem
    grid = cells(100, 1, -0.5, 0.5, 0, 1)
    model = TwoFluidEMHD(grid=grid)
    source = sources(twoFluidDivClean)
    # Set initial state
    initFuncObj = initialFunc(grid, model)
    initFunc = initFuncObj.twoFluidBrioWu


    # Set up simulation
    sim = simulation(initFuncObj, initFunc, grid,
                     model, backEulerRK2, outflow, fluxSplitting, source=source,
                     cfl=0.4)

    print("Running simulation")
#    sim.runSim(0.1)
#    sim.plotPrimHeatmaps() 
    sim.runSim(0.1)



    timeTotal = time.time() - timeStart
    print("Run time: {:.0f}m {:.1f}s".format(timeTotal // 60, timeTotal%60 ))



