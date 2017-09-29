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
        self.prims = None
        self.aux = None
        self.model = model
        self.grid = grid
        self.primL = primL
        self.primR = primR

        try:
            self.g = model.g
        except AttributeError:
            pass


    def RiemannProbGeneral(self):
        assert((self.primL is not None) and (self.primR is not None)), \
        "initialFunc must have left/right primitive states for relEulerEqns"
        x, y = self.grid.coordinates()
        primvars = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))
        
        
        # Discontinuous in x-direction
        for i, var in enumerate(primvars):
            for j in range(x.shape[0]):
                var[j, :] = np.where(x<=0, self.primL[i], self.primR[i])
            
        # Discontinuous in y-direction
#        for i, var in enumerate(primvars):
#            for j in range(y.shape[0]):
#                var[:, j] = np.where(y<=0, primL[i], primR[i])

        self.prims = primvars
        cons, self.aux, alpha = self.model.getConsFromPrims(primvars)
        return cons

    def Migone(self):
        """
        A two dimensional hydrodynamics test problem. Explained in
        http://flash.uchicago.edu/~jbgallag/2012/flash4_ug/node34.html#SECTION081110000000000000000
        """
        x, y = self.grid.coordinates()
        prims = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))
                
        nx = x.shape[0]
        ny = y.shape[0]
        
        for i in range(nx):
            for j in range(ny):
                if i < nx/2 and j < ny/2:
                    prims[0, i, j] = 0.5
                    prims[3, i, j] = 1.0
                elif i >= nx/2 and j < ny/2:
                    prims[0, i, j] = 0.1
                    prims[3, i, j] = 1.0
                    prims[2, i, j] = 0.99
                elif i < nx/2 and j >= ny/2:
                    prims[0, i, j] = 0.1
                    prims[3, i, j] = 1.0
                    prims[1, i, j] = 0.99
                elif i >= nx/2 and j >= ny/2:
                    prims[0, i, j] = 5.477875e-3
                    prims[3, i, j] = 2.762987e-3
                    
        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)
        return cons
    












