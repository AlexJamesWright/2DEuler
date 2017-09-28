"""
initialFunctions.py

script that stores the functional form of the initial fields
"""

import numpy as np


class initialFunc(object):
    def __init__(self, grid, model=None, tau=1, primL=None, primR=None):
        """
        Class that stores the functional forms of the initial field

        Parameters
        ----------
        grid: object
            A cell object, simulation.cells, containing the information about
            the system's grid construction
        model: object (optional)
            simulation.model object containing onfo about the type of problem,
            i.e. the equations we are solving (Advection, Euler eqns etc)
        tau: float (optional)
            The relaxation time of the source term - only meaningful if there
            is a source term
        primL: array of float (optional)
            (Nvars,) The initial data of the primative variables on the left
            side of the system
        primR: array of float (optional)
            (Nvars,) The initial data of the primative variables on the right
            side of the system
        """
        self.tau = tau
        self.primL = primL
        self.primR = primR
        self.prims = None
        self.aux = None
        self.model = model
        self.grid = grid

        try:
            self.g = model.g
        except AttributeError:
            pass


    def RiemannProbGeneral(self):
        assert((self.primL is not None) and (self.primR is not None)), \
        "initialFunc must have left/right primitive states for relEulerEqns"
        x, y = self.grid.coordinates()
        dx = self.grid.deltaX
        primvars = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))
        
        
        # Discontinuous in x-direction
        for i, var in enumerate(primvars):
            for j in range(y.shape[0]):
                var[:, j] = np.where(x<=0, self.primL[i], self.primR[i])

        # Discontinuous in y-direction
#        for i, var in enumerate(primvars):
#            for j in range(x.shape[0]):
#                var[j, :] = np.where(y<=0, self.primL[i], self.primR[i])


        self.prims = primvars
        cons, self.aux, alpha = self.model.getConsFromPrims(primvars, dx)
        return cons

