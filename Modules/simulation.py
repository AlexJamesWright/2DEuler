"""
simulation.py

class to store data about hydro simulations
"""
import numpy as np
import pylab as plt

class simulation(object):
    """
    Parameters
    ----------
    initialFunc: callable
        The functional form of the initial field
    cells: class
        Contains information on all the cells' geometry
    model: callable
        The functional form of the field e.g. advection, burgers' etc.
    timeEv: callable
        The method of timestepping desired e.g. Euler, RK etc.
    bcs: callable
        Function that resets all data according to specified boundary
        conditions
    F: callable
        The form of the flux approximation e.g. upwind, lax-friedrichs etc
        (basically my version of NFluxFunc(..) - NFluxFunc(..) )
        script stored in fluxApprox.py
    restruct: callable
        The form of the slope reconstruction e.g. constant, minmod, weno3 etc
    cfl: float (optional)
        Courant–Friedrichs–Lewy condition number
        cfl = deltaT/deltaX


    Example call:
        sim = simulation(initialFunctions.sin, cells.cells(100),
                         model.advection, timeEv.RK2, bcs.periodic,
                         fluxApprox.upwind, slopeReconstruction.linear)
    """

    def __init__(self, initFuncObj, initialFunc, cells, model, timeEv, bcs, F, restruct=None,
                 source=None, cfl=0.5):
        self.model = model                   # Advection/Burgers etc.
        self.Nvars = self.model.Nvars                # Number of variables in q
        self.flux = self.model.flux            # call with direction=0 for x, =1 for y 
        self.cells = cells                      # Grid
        self.restruct = restruct                # Slope reconstruction (eg weno)
        self.sourceClass = source               # Class containing sources
        self.source = source.sourceF            # Functional from of source
        self.F = F                              # Flux approx (eg upwind/LaxF)
        self.bcs = bcs                          # Boundary Cond. (eg. peridoc)
        self.timeEv = timeEv                    # Time evolution (eg Euler, RK)
        self.cfl = cfl                          # Value for courant number
        self.deltaX = cells.deltaX              # Each cell's width
        self.deltaY = cells.deltaY
        self.deltaT = self.cfl * self.deltaX    # Simulation timestep
        self.coordinates = cells.coordinates()  # All grid coordinates
        self.initFuncObj = initFuncObj          # The initial function object
        self.initialFunc = initialFunc          # Functional form of initial field
        self.q0 = initialFunc()                 # Initial field value: conserved variables
        self.prims0 = self.initFuncObj.prims    # Primative variables for whole system
        self.q0 = self.q0.reshape((self.Nvars, self.cells.nx \
                                   + 2*self.cells.Nghosts, self.cells.ny \
                                   + 2*self.cells.Nghosts))
        self.q = self.q0.copy()                 # Current field value
        self.prims = self.prims0
        self.getPrims = self.updatePrims()      # Callable, given conserved vars return prims and aux if exist, None otherwise
        self.aux0 = self.initFuncObj.aux
        self.aux = self.aux0
        self.c = None                           # Sound speed in each cell
        self.t = 0                              # Current time
        self.iters = 0                          # Number of timesteps iterated
        self.consLabels = self.model.consLabels # Conserved Variable labels
        self.primLabels = self.model.primLabels # Primitive Variable labels
        self.auxLabels = self.model.auxLabels   # Auxilliary variable labels
        self.alpha = 1                          # The maximum wave speed, max(abs(u + c))

    def updateTime(self, endTime):
        """
        Parameters
        ----------
        endTime: float
            The maximum runtime of the simulation

        updateTime increments the simulation by the initialised timestep, and
        updates the field values of all the cells
        """
        self.deltaT = self.cfl / (1/self.deltaX + 1/self.deltaY)
        if self.iters < 5:
            self.deltaT *= 0.1
        if self.t + self.deltaT > endTime:
            self.deltaT = endTime - self.t
        self.q = self.timeEv(self, self.q)
        self.q = self.bcs(self.q, self.cells)
        self.prims, self.aux, self.alpha = self.getPrims(self.q)
        self.t += self.deltaT
        self.iters += 1



    def runSim(self, endTime):
        """
        Parameters
        ----------
        endTime: float
            The maximum run time of the simulation

        runSim continually updatesTime until the endTime has been reached
        """

        self.prims, self.aux, self.alpha = self.getPrims(self.q)
        self.alpha=1
        while self.t < endTime:
            print("t = {}".format(self.t))
            self.updateTime(endTime)

    def updatePrims(self):
        """
        Returns the function that generates the primative and auxiliary variables
        if they are defined, else the function thats returned will always
        return None
        """

        if (self.prims is None):
            def f(q):
                return None, None, 1 # Alpha must not be None
            return f
        else:
            def g(q):
                prims, aux, alpha = self.model.getPrimitiveVars(q, self)
                prims = self.bcs(prims, self.cells)
                aux = self.bcs(aux, self.cells)
                return prims, aux, alpha

            return g


    def reset(self):
        """
        Resets all data to initial values
        """
        self.q = self.q0
        self.prims = self.prims0
        self.aux = None
        self.iters = 0
        self.t = 0



    ################################################################
    ###################     Plotting Tools     #####################
    ################################################################
    def plotConservedVars(self):
        """
        Plots the final field for the conserved variables of the system
        """
        assert(self.Nvars > 1), "Only appropriate for systems with more than one fluid"
        for i in range(self.Nvars):
            plt.figure()
            ymin = np.min(self.q[i, self.cells.Nghosts:-self.cells.Nghosts])
            ymax = np.max(self.q[i, self.cells.Nghosts:-self.cells.Nghosts])
            dy = ymax - ymin
            ylower = ymin - 0.05 * dy
            yupper = ymax + 0.05 * dy
            xs, ys = self.cells.realCoordinates()
            plt.plot(xs,
                    self.q[i, self.cells.Nghosts:-self.cells.Nghosts])
            #py.title(r'Time Evolution for fluid {}: $t = {}$'.format(self.consLabels[i], self.t))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$q_{}(x)$'.format(i+1))
            plt.ylim((ylower, yupper))
            plt.legend(loc='upper right')
#            plt.savefig('Figures/ViolateCharacteristicSSP3{}.pdf'.format(self.consLabels[i]))
            plt.show()

    def plotPrimitives(self):
        """
        Plots the final field for the primitive variables of the system
        """
        Nx, Ny = self.prims[0].shape
        Ng = self.cells.Nghosts
        for i in range(self.model.Nprims):
            plt.figure()
            ymin = np.min(self.prims[i, Ng:-Ng, Ny//2])
            ymax = np.max(self.prims[i, Ng:-Ng, Ny//2])
            dy = ymax - ymin
            ylower = ymin - 0.05 * dy
            yupper = ymax + 0.05 * dy
            xs, ys = self.cells.realCoordinates()
            plt.plot(xs,
                    self.prims[i, Ng:-Ng, Ny//2])
            plt.title(r'Time Evolution for {}: $t = {}$'.format(self.primLabels[i], self.t))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$q_{}(x)$'.format(i+1))
            plt.ylim((ylower, yupper))
            plt.legend(loc='lower center', fontsize=10)
#            py.savefig('Figures/GR2MHD{}.pdf'.format(self.primLabels[i]))
            plt.show()
            
    def plotPrimHeatmaps(self, initial=0, color=None):
        from matplotlib import cm
        Ng = self.cells.Nghosts
        xs, ys = self.cells.realCoordinates()
        for i in range(self.prims.shape[0]):
            fig = plt.figure()
            plotPrims = self.prims[i, Ng:-Ng, Ng:-Ng]
            if initial:
                plotPrims = self.prims0[i, Ng:-Ng, Ng:-Ng]
            if color == None:
                color = cm.plasma
            surf = plt.imshow(plotPrims, cmap=color, interpolation='nearest')
            plt.title(r'Time Evolution for {}: $t = {}$'.format(self.primLabels[i], self.t))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.legend()
            plt.show()
            
    def plotPrimSurface(self):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        
        for i in range(self.prims.shape[0]):
            Ng = self.cells.Nghosts
            xs, ys = self.cells.realCoordinates()
            var = self.prims[i, Ng:-Ng, Ng:-Ng]
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x, y = np.meshgrid(xs, ys)
            surf = ax.plot_surface(x, y, var, linewidth=0, cmap=cm.gist_rainbow)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.w_zaxis.set_major_locator(LinearLocator(6))
            plt.xlabel(r'$x$')
            plt.ylabel(r'$y$')
            plt.title(r'Solution for {}'.format(self.primLabels[i]))
            plt.show()

    def plotLogDensity(self):
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        
        Ng = self.cells.Nghosts
        xs, ys = self.cells.realCoordinates()
        var = np.log10(self.prims[0, Ng:-Ng, Ng:-Ng])
        fig = plt.figure()
        surf = plt.imshow(var, cmap=cm.coolwarm)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(r'Solution for {}'.format(self.primLabels[0]))
        plt.show()
