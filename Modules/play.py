"""
play.py

script to test, play, and explore the functionalily of simulation.py
"""


from simulation import simulation
from initialFunctions import initialFunc
from cells import cells
from model import SRMHDClass
from timeEv import RK2
from boundaryConditions import periodic
from fluxApprox import fluxSplitting
from sourceTerms import sources, noSource
import numpy as np
import time


if __name__ == '__main__':
    timeStart = time.time()

    # Set up problem
    grid = cells(50, 50, 0, 1, 0, 1)
    model = SRMHDClass(grid=grid, g=5/3)
    source = sources(noSource, tau=None)
    # Set initial state
    primL = np.array([1, 0, 0, 100])
    primR = np.array([0.125, 0, 0, 0.1])
    initFuncObj = initialFunc(grid, model, source.tau, primL, primR)
    initFunc = initFuncObj.OTVortex


    # Set up simulation
    sim = simulation(initFuncObj, initFunc, grid,
                     model, RK2, periodic, fluxSplitting, source=source,
                     cfl=0.2)

    print("Running simulation")
    sim.runSim(0.1)
    sim.plotPrimHeatmaps() 


    timeTotal = time.time() - timeStart
    print("Run time: {:.0f}m {:.1f}s".format(timeTotal // 60, timeTotal%60 ))


