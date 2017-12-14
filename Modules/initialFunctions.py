"""
initialFunctions.py

script that stores the functional form of the initial fields
"""

import numpy as np
import scipy


class initialFunc(object):
    def __init__(self, grid, model=None, direction=0, primL=None, primR=None):
        """
        Class that stores the functional forms of the initial field

        Parameters
        ----------
        grid : object
            A cell object, simulation.cells, containing the information about
            the system's grid construction
        model : object (optional)
            simulation.model object containing onfo about the type of problem,
            i.e. the equations we are solving (Advection, Euler eqns etc).
            Defaults to None.
        direction : int (optional)
            Where possible, defines the direction of the initial set up. E.g. for
            twoFluidBrioWu sets the discontinuity along the dir axis, where
            dir = [0, 1] = [x, y]. Defaults to x-axis
        tau: float (optional)
            The relaxation time of the source term - only meaningful if there
            is a source term
        primL : array of float (optional)
            (Nvars,) The initial data of the primative variables on the left
            side of the system. Defaults to None.
        primR : array of float (optional)
            (Nvars,) The initial data of the primative variables on the right
            side of the system.  Defaults to None.
        """
        self.prims = None
        self.aux = None
        self.model = model
        self.direction = direction
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
        assert(self.model.Nprims == 9), "Migone only valid for single-fluid model"
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



    def OTVortexSingle(self):
        """
        Orszang-Tang Vortex: A two dimensional
        magnetohydrodynamic test problem. In the relativistic case we have to
        ensure that the velocities do not exceed 1.
        See http://flash.uchicago.edu/site/flashcode/user_support/flash_ug_devel/node178.html
        """
        assert(self.grid.xmin == 0.0 and self.grid.xmax == 1.0), "X E [0, 1]"
        assert(self.grid.ymin == 0.0 and self.grid.ymax == 1.0), "Y E [0, 1]"
        assert(self.model.Nprims == 9), "OTVortexSingle only valid for single-fluid model"


        x, y = self.grid.coordinates()
        prims = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))



        prims[0, :, :] = self.model.g**2 / 4 / np.pi
        prims[4, :, :] = self.model.g / 4 / np.pi

        for i, xelem in enumerate(x):
            prims[2, i, :] = 0.5*np.sin(2 * np.pi * xelem)
            prims[6, i, :] = np.sin(4 * np.pi * xelem) / np.sqrt(4*np.pi)

        for j, yelem in enumerate(y):
            prims[1, :, j] = - 0.5*np.sin(2 * np.pi * yelem)
            prims[5, :, j] = - np.sin(2 * np.pi * yelem) / np.sqrt(4*np.pi)

        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)

        return cons


    def OTVortexTwoFluid(self):
        """
        Orszang-Tang Vortex: A two dimensional magnetohydrodynamic test problem.
        Specifically, this is the set up for the two fluid model TwoFluidEMHD taken
        from Amano 16 (and refs therein).
        See "A second-order divergence-constrained multidimensional numerical
        scheme for relativistic two-fluid electrodynamics"
        """
        assert(self.grid.xmin == 0.0 and self.grid.xmax == 1.0), "X E [0, 1]"
        assert(self.grid.ymin == 0.0 and self.grid.ymax == 1.0), "Y E [0, 1]"
        assert(self.model.Nprims == 16), "OTVortexTwoFluid only valid for two-fluid model"
        x, y = self.grid.coordinates()

        prims = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims


        rho1[:] = self.model.g**2 / 4 / np.pi
        rho2[:] = self.model.g**2 / 4 / np.pi
        p1[:] = self.model.g / 4 / np.pi
        p2[:] = self.model.g / 4 / np.pi

        for i, xelem in enumerate(x):
            vy1[i] = 0.5 * np.cos(2 * np.pi * xelem)
            vy2[i] = 0.5 * np.cos(2 * np.pi * xelem)
            By[i] = np.sin(4 * np.pi * xelem) / np.sqrt(4*np.pi)

        for j, yelem in enumerate(y):
            vx1[:, j] = - 0.5 * np.sin(2 * np.pi * yelem)
            vx2[:, j] = - 0.5 * np.sin(2 * np.pi * yelem)
            Bx[:, j] = - np.sin(2 * np.pi * yelem) / np.sqrt(4*np.pi)

        Ex = vy1 * Bz - vz1 * By
        Ey = vz1 * Bx - vx1 * Bz
        Ez = vx1 * By - vy1 * Bx

        prims[:] = rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez


        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)

        return cons


    def twoFluidStatic(self):
        """
        An equilibrium set up, all vars are constant and should result in 
        zero flux.
        """
        assert(self.model.Nprims == 16), "twoFluidStatic only valid for two-fluid model"
                
        x, y = self.grid.coordinates()

        prims = np.zeros((self.model.Nprims, x.shape[0], y.shape[0]))
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims
        
        rho1[:] = 1.0
        rho2[:] = 1.0
        p1[:] = 1.0
        p2[:] = 1.0
        Bx[:] = 1.0
        By[:] = 1.0
        Bz[:] = 1.0
        Ex[:] = 1.0
        Ey[:] = 1.0
        Ez[:] = 1.0
        
        

        prims[:] = rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez

        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)

        return cons
    
    def twoFluidBrioWu(self):
        """
        Generalized  brio wu shock tube test
        """
        assert(self.grid.xmin == -0.5 and self.grid.xmax == 0.5), "X E [-0.5, 0.5]"
        assert(self.model.Nprims == 16), "twoFluidBrioWu only valid for two-fluid model"
                
        x, y = self.grid.coordinates()
        Nx, Ny = x.shape[0], y.shape[0]
        prims = np.zeros((self.model.Nprims, Nx, Ny))
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims
        
        endX = Nx - 1
        endY = Ny - 1
        facX = 1
        facY = 1
        if (self.direction == 0):
            facX = 2
            lBx = 0.5
            rBx = 0.5
            lBy = 1.0
            rBy = -1.0
        else:
            facY = 2
            lBy = 0.5
            rBy = 0.5
            lBx = 1.0
            rBx = -1.0
            
        
        for i in range(Nx//facX):
            for j in range(Ny//facY):
                rho1[i, j] = 0.5
                rho2[i, j] = 0.5
                p1[i, j] = 1.0
                p2[i, j] = 1.0
                Bx[i, j] = lBx
                By[i, j] = lBy
                
                rho1[endX - i, endY - j] = 0.075
                rho2[endX - i, endY - j] = 0.075
                p1[endX - i, endY - j] = 0.1
                p2[endX - i, endY - j] = 0.1
                Bx[endX - i, endY - j] = rBx
                By[endX - i, endY - j] = rBy
        

        prims[:] = rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez

        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)

        return cons
    
    def twoFluidCurrentSheet(self):
        """
        Self similar current sheet for EP plasma
        """
        assert(self.grid.xmin == -1.5 and self.grid.xmax == 1.5), "X E [-1.5, 1.5]"
        assert(self.model.Nprims == 16), "twoFluidCurrentSheet only valid for two-fluid model"
                
        x, y = self.grid.coordinates()
        Nx, Ny = x.shape[0], y.shape[0]
        prims = np.zeros((self.model.Nprims, Nx, Ny))
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims
        
        RHO = 1.0
        B0 = 1.0
        P = 50.0
        
        tmp1 = (B0 / (self.model.mu1 * RHO * np.sqrt(np.pi / self.model.sig))) * np.exp(- x**2 * self.model.sig / 4.0)
        tmp2 = (B0 / (self.model.mu2 * RHO * np.sqrt(np.pi / self.model.sig))) * np.exp(- x**2 * self.model.sig / 4.0)
        for j in range(Ny):
            vz1[:, j] = (tmp1 / np.sqrt(1 - tmp1**2))
            vz2[:, j] = tmp2 / np.sqrt(1 - tmp2**2)
            By[:, j] = B0 * np.sign(x)*scipy.special.erf(0.5 * np.sqrt(self.model.sig * x ** 2))


        rho1[:] = RHO / 2.0
        p1[:] = P / 2.0
        rho2[:] = RHO / 2.0 
        p2[:] = P / 2.0
        
        
        
        prims[:] = rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez
        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)
        
        return cons
    
    def twoFluidCPAlfven(self):
        assert(self.model.Nprims == 16), "twoFluidCPAlfven only valid for two-fluid model"
        assert(self.grid.xmin == 0 and self.grid.xmax == 8 * np.pi), "boundaries should be x E [0, 8Pi]"
#        assert(self.grid.ny == 0), "Must be one dimensional"
        assert(abs(self.model.mu1) == np.sqrt(1.04) and abs(self.model.mu2)==np.sqrt(1.04)), "Charge mass ratio should be sqrt(1.04)"
        
        x, y = self.grid.coordinates()
        Nx, Ny = x.shape[0], y.shape[0]
        prims = np.zeros((self.model.Nprims, Nx, Ny))
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims

        
        B0 = 1.04
        omegaBar1 = -np.sqrt(1.04)
        omegaBar2 = - omegaBar1
        kx = 1.0/4.0
        omega = 5.63803828148e-1
        Wp = 5.19940020571e-6 + 1
        We = 6.68453076522e-5 + 1
        Tom = 10**-2
        xsi = 0.01
        U1 = -xsi * omega * omegaBar1 / (kx * (omega + omegaBar1/We))
        U2 = -xsi * omega * omegaBar2 / (kx * (omega + omegaBar2/Wp))
        
        for j in range(Ny):
            phi = kx * x
            
            # Density and pressure
            rho1[:, j] = 1.0 / We
            p1[:, j] = Tom * rho1[:, j]
            rho2[:, j] = 1.0 / Wp
            p2[:, j] = Tom * rho2[:, j]
            
            # Magnetic fields
            Bx[:, j] = B0 
            By[:, j] = xsi * B0 * np.cos(phi)
            Bz[:, j] = -xsi * B0 * np.sin(phi)
            
            # Electric fields
            Ey[:, j] = - (omega / kx) * xsi * B0 * np.sin(phi)
            Ez[:, j] = - (omega / kx) * xsi * B0 * np.cos(phi)
            
            # vy1, vz1, vy2, vz2
            vy1[:, j] = U1 * np.cos(phi) / We
            vz1[:, j] = - U1 * np.sin(phi) / We
            vy2[:, j] = U2 * np.cos(phi) / Wp
            vz2[:, j] = - U2 * np.sin(phi) / Wp
            
            
        prims[:] = rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, vz2, p2, Bx, By, Bz, Ex, Ey, Ez
        self.prims = prims
        cons, self.aux, alpha = self.model.getConsFromPrims(prims)
        
        
        
        return cons
        
        
        