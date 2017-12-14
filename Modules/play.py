"""
play.py

script to test, play, and explore the functionalily of simulation.py
"""


from simulation import simulation
from initialFunctions import initialFunc
from cells import cells
from model import TwoFluidEMHD
from timeEv import backEulerRK2, eulerSplitRK2, eulerSplitRK3, RK2
from boundaryConditions import periodic, outflow
from fluxApprox import fluxSplitting
from sourceTerms import sources, twoFluidDivClean
import numpy as np

import time


if __name__ == '__main__':
    timeStart = time.time()

    # Set up problem
    grid = cells(200, 1, 0, 8*np.pi, -1.5, 1.5)
    model = TwoFluidEMHD(grid=grid, g=4.0/3.0, sig=1e16, mu1=-np.sqrt(1.04), mu2=np.sqrt(1.04))
    source = sources(twoFluidDivClean, model.mu1, model.mu2)
    # Set initial state
    initFuncObj = initialFunc(grid, model, direction=0)
    initFunc = initFuncObj.twoFluidCPAlfven


    # Set up simulation
    sim = simulation(initFuncObj, initFunc, grid,
                     model, eulerSplitRK2, outflow, fluxSplitting, source=source,
                     cfl=0.1)
    

    print("Running simulation")
#    sim.runSim(0.0)
#    sim.updateTime(1)
    sim.updateTime(1)
    sim.plotTwoFluidCPAlfvenWaveAgainstExact()

    timeTotal = time.time() - timeStart
    print("Run time: {:.0f}m {:.1f}s".format(timeTotal // 60, timeTotal%60 ))




