import numpy as np
from scipy.optimize import newton, brentq, fsolve
from weno import weno_upwind
import warnings
import cells

warnings.filterwarnings('ignore', 'The iteration is not making good progress')


#############################################################################
##########                        TFEMHD                         ############        
#############################################################################

class TwoFluidEMHD(object):
    def __init__(self, grid, g=5/3, mu1=-1, mu2=1, sig=100, kappa=100):
        """
        Two Fluid ElectroMagnetoHydroDynamics
        The two fluid model for plasmas in the special relativistic limit. Model
        is taken from Amano '16 "A second-order divergence-constrained 
        multidimensional numerical scheme for relativistic two-fluid 
        electrodynamics", although here we implement divergence cleaning to constrain
        the conditions set by maxwells equations. The divergence cleaning method
        is taken from Kiki's Thesis (and refs) for resistive SRMHD. 
        
        Cons = (D, Sx, Sy, Sz, tau,
                Dbar, Sbarx, Sbary, Sbarz, tauBar,
                Bx, By, Bz, Ex, Ey, Ez,
                psi, phi)
        
        where we have implied the summatation over species 1 and 2 (electrons 
        and protons), such that D = D1 + D2 etc. The bars refer to the weighted
        sum of the respective variables (weighted by the charge/mass ratio).
        
        D       = rho * W
        Si      = rho * h * W**2 * vi + EcrossBi
        tau     = rho * h * W - p + 0.5 * (Esq + Bsq)
        Dbar    = mu * rho * W
        Sbari   = mu * rho * h * W**2 * vi
        tauBar  = mu * rho * h * W - mu * p
        
        Prims = (rho1, vx1, vy1, vz1, p1,
                 rho2, vx2, vy2, vz2, p2,
                 Bx, By, Bz, Ex, Ey, Ez)
        
        Aux = (h1, W1, e1, vsq1, Z1, vE1, D1, Stildex1, Stildey1, Stildez1, tauTilde1
               h2, W2, e2, vsq2, Z2, vE2, D2, Stildex2, Stildey2, Stildez2, tauTilde2
               Bsq, Esq, 
               Jx, Jy, Jz, 
               Stildex, Stildey, Stildez, 
               tauTilde
               rhoCh, rhoCh0,
               ux, uy, uz, W)
        
        where:
            D1          = (Dbar - mu2 * D) / (mu1 - mu)             etc for D2
            tauTilde    = tau - 0.5 * (Esq - Bsq)
            tauTilde1   = (tauBar - mu2 * tauTilde) / (mu1 - mu2)   etc for tauTilde2
            Stildei     = Si - EcrossB
            Stildei1    = (Sbari - mu2 * Stildei) / (mu1 - mu)      etc for Stildei2
            vE1         = vx1 * Ex + vy1 * Ey + vz1 * Ez            etc for vE2
            Ji          = mu1 * rho1 * W1 * vi1 + mu2 * rho2 * W2 * vi2
            rhoCh       = mu1 * rho1 * W1 + mu2 * rho2 * W2
            W           = (mu1*mu1*rho1*W1 + mu2*mu2*rho2*W2) / (mu1*mu1*rho1 + mu2*mu2*rho2)
            ui          = (mu1*mu1*rho1*W1*v1i + mu2*mu2*rho2*W2*v2i) / (mu1*mu1*rho1 + mu2*mu2*rho2)
            rhoCh0      = W * rhoCh - numpy.dot(J, u)
            Z1          = rho1 * h1 * W1**2
            
        Good luck.
        """
        
        self.consLabels = [r'$D$', r'$S_x$', r'$S_y$', r'$S_z$', r'$\tau$',
                           r'$\bar{D}$', r'$\bar{S}_x$', r'$\bar{S}_y$', r'$\bar{S}_z$', 
                           r'$\bar{\tau}$', r'$B_x$', r'$B_y$', r'$B_z$', r'$E_x$', 
                           r'$E_y$', r'$E_z$', r'$\psi$', r'$\phi$']
        self.primLabels = [r'$\rho_1$', r'$v_{x1}$', r'$v_{y1}$', r'$v_{z1}$', r'$p_1$',
                            r'$\rho_2$', r'$v_{x2}$', r'$v_{y2}$', r'$v_{z2}$', r'$p_2$', 
                            r'$B_x$', r'$B_y$', r'$B_z$', r'$E_x$', r'$E_y$', r'$E_z$']
        self.auxLabels = [r'$h_1$', r'$W_1$', r'$e_1$', r'$v^2_1$', r'$Z_1$', r'$vE_1$', r'$D_1$', r'$\tilde{S}_{x1}$', r'$\tilde{S}_{y1}$', r'$\tilde{S}_{z1}$', r'$\tilde{\tau}_{1}$',
                          r'$h_2$', r'$W_2$', r'$e_2$', r'$v^2_2$', r'$Z_2$', r'$vE_2$', r'$D_2$', r'$\tilde{S}_{x2}$', r'$\tilde{S}_{y2}$', r'$\tilde{S}_{z2}$', r'$\tilde{\tau}_{2}$',
                          r'$B^2$', r'$E^2$', 
                          r'$J_{x}$', r'$J_{y}$', r'$J_{z}$', 
                          r'$\tilde{S}_x$', r'$\tilde{S}_z$', r'$\tilde{S}_z$', 
                          r'$\tilde{\tau}$',
                          r'$\rho_{ch}', r'$\rho_{ch_0}$',
                          r'$u_x$', r'$u_z$', r'$u_y$', r'$W$']
        self.grid = grid
        self.g = g
        self.mu1 = mu1
        self.mu2 = mu2
        self.sig = sig
        self.kappa = kappa
        self.Nvars = 18
        self.Nprims = 16
        self.Naux = 37
        self.flux = self.fluxFunc
        
    def fluxFunc(self, q, sim):
        pass
    
    def getPrimitiveVars(self, q, sim):
        """
        Given the conserved variables, returns the corresponding values for the 
        primitive and aux vars. 
        First subtract the EM fields from the momentum and energy, split up the
        cons vars into contributions from each fluid species, and the proceed in
        the normal relativistic hydrodynamic fashion.
        """
        # Short hand
        mu1 = self.mu1
        mu2 = self.mu2
        Nx, Ny =  q[0, :, :].shape
        
        # Initialize variables
        aux = np.zeros((self.Naux, Nx, Ny))
        prims = np.zeros((self.Nprims, Nx, Ny))
        # Aux
        h1, W1, e1, vsq1, Z1, vE1, D1, Stildex1, Stildey1, Stildez1, tauTilde1, \
        h2, W2, e2, vsq2, Z2, vE2, D2, Stildex2, Stildey2, Stildez2, tauTilde2, \
        Bsq, Esq, \
        Jx, Jy, Jz, \
        Stildex, Stildey, Stildez, \
        tauTilde, \
        rhoCh, rhoCh0, \
        ux, uy, uz, W = aux[:]
        # Prims
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2,\
        vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims[:]

        # Cons vars
        D, Sx, Sy, Sz, tau, Dbar, Sbarx, Sbary, Sbarz, \
        tauBar, Bx, By, Bz, Ex, Ey, Ez, psi, phi = q[:]
        
        Bsq = Bx**2 + By**2 + Bz**2
        Esq = Ex**2 + Ey**2 + Ez**2
        
        # Remove EM contribution
        Stildex = Sx - (Ey * Bz - Ez * By)
        Stildey = Sy - (Ez * Bx - Ex * Bz)
        Stildez = Sz - (Ex * By - Ey * Bx)
        tauTilde = tau - 0.5 * (Esq + Bsq)
        
        # Split the fluid into its constituent species
        D1 = (Dbar - mu2 * D) / (mu1 - mu2)
        D2 = (Dbar - mu1 * D) / (mu2 - mu1)
        Stildex1 = (Sbarx - mu2 * Stildex) / (mu1 - mu2)
        Stildey1 = (Sbary - mu2 * Stildey) / (mu1 - mu2)
        Stildez1 = (Sbarz - mu2 * Stildez) / (mu1 - mu2)
        Stildex2 = (Sbarx - mu1 * Stildex) / (mu2 - mu1)
        Stildey2 = (Sbary - mu1 * Stildey) / (mu2 - mu1)
        Stildez2 = (Sbarz - mu1 * Stildez) / (mu2 - mu1)
        Stilde1sq = Stildex1**2 + Stildey1**2 + Stildez1**2
        Stilde2sq = Stildex2**2 + Stildey2**2 + Stildez2**2
        tauTilde1 = (tauBar - mu2 * tauTilde) / (mu1 - mu2)
        tauTilde2 = (tauBar - mu1 * tauTilde) / (mu2 - mu1)
        
        
        # We now have everything we need
        def residual(Z, StildesqFluid, DFluid, tauTildeFluid, i, j):
            """
            Function to minimize to determine value of guess=Z=rho * h * W * W
            Parameters
            ----------
                Z: float
                    Estimate of solution, Z = rho * h * W**2
                StildesqFluid: float
                    Value of Stildesq for this fluid
                DFluid: float
                    Value of D for this fluid
                tauTildeFluid: float
                    Value of tauTilde for this fluid
            """
            vsq = StildesqFluid / Z**2
            if (vsq >= 1):
                return 1e6
            W = 1 / np.sqrt(1 - vsq)
            rho = DFluid / W
            h = Z / (rho * W**2)
            p = (self.g - 1) * (h - rho) / self.g            
            if Z < 0 or rho < 0 or p < 0 or W < 1 or h < 1:
                return 1e6
            else:
                resid = (1 - (self.g - 1)/(W**2*self.g)) * Z + \
                           ((self.g - 1)/(W*self.g) - 1)*DFluid - tauTildeFluid
                return resid
        
        
        # Loop over all domain cells
        for i in range(Nx):
            for j in range(Ny):
                # Fluid 1
                Z1[i, j] = newton(residual, x0=sim.aux[4, i, j]*0.95, args=(Stilde1sq[i, j], D1[i, j], tauTilde1[i, j], i, j))
                # Fluid 2
                Z2[i, j] = newton(residual, x0=sim.aux[15, i, j]*0.95, args=(Stilde2sq[i, j], D2[i, j], tauTilde2[i, j], i, j))




        return Z1, Z2
        
    
    def getConsFromPrims(self, prims):
        """
        Generates the conserved and auxilliary vectors from the primitive values.
        Used at start of sim, initial conditions are set in primitives, need to
        transformed. 
        """
        mu1 = self.mu1
        mu2 = self.mu2
        Nx, Ny =  prims[0, :, :].shape
        cons = np.zeros((self.Nvars, Nx, Ny))
        aux = np.zeros((self.Naux, Nx, Ny))
        alpha = 1
        psi = np.zeros_like(prims[0])
        phi = np.zeros_like(prims[0])
        
        rho1, vx1, vy1, vz1, p1, rho2, vx2, vy2, \
        vz2, p2, Bx, By, Bz, Ex, Ey, Ez = prims
        
        vsq1        = vx1**2 + vy1**2 + vz1**2
        vsq2        = vx2**2 + vy2**2 + vz2**2
        W1          = 1 / np.sqrt(1 - vsq1)
        W2          = 1 / np.sqrt(1 - vsq2)
        EcrossBx    = Ey * Bz - Ez * By
        EcrossBy    = Ez * Bx - Ex * Bz
        EcrossBz    = Ex * By - Ey * Bx
        Esq         = Ex**2 + Ey**2 + Ez**2
        Bsq         = Bx**2 + By**2 + Bz**2
        e1          = p1 / (rho1 * (self.g - 1))
        e2          = p2 / (rho2 * (self.g - 1))
        h1          = 1 + e1 + p1 / rho1
        h2          = 1 + e2 + p2 / rho2
        Z1          = rho1 * h1 * W1**2
        Z2          = rho2 * h2 * W2**2
        vE1         = vx1*Ex + vy1*Ey + vz1*Ez
        vE2         = vx2*Ex + vy2*Ey + vz2*Ez
        Jx          = mu1 * rho1 * W1 * vx1 + mu2 * rho2 * W2 * vx2
        Jy          = mu1 * rho1 * W1 * vy1 + mu2 * rho2 * W2 * vy2
        Jz          = mu1 * rho1 * W1 * vz1 + mu2 * rho2 * W2 * vz2
        rhoCh       = mu1 * rho1 * W1 + mu2 * rho2 * W2
        W           = (mu1**2 * rho1 * W1 + mu2**2 * rho2 * W2) / \
                      (mu1**2 * rho1 + mu2**2 * rho2)
        ux          = (mu1**2 * rho1 * W1 * vx1 + mu2**2 * rho2 * W2 * vx2) / \
                      (mu1**2 * rho1 + mu2**2 * rho2)
        uy          = (mu1**2 * rho1 * W1 * vy1 + mu2**2 * rho2 * W2 * vy2) / \
                      (mu1**2 * rho1 + mu2**2 * rho2)
        uz          = (mu1**2 * rho1 * W1 * vz1 + mu2**2 * rho2 * W2 * vz2) / \
                      (mu1**2 * rho1 + mu2**2 * rho2)
        rhoCh0      = W * rhoCh - (Jx*ux + Jy*uy + Jz*uz)
        D1          = rho1 * W1
        D2          = rho2 * W2
        D           = D1 + D2
        Sx          = Z1 * vx1 + Z2 * vx2 + EcrossBx
        Sy          = Z1 * vy1 + Z2 * vy2 + EcrossBy
        Sz          = Z1 * vz1 + Z2 * vz2 + EcrossBz
        tau         = Z1 - p1 + Z2 - p2 + 0.5 * (Esq + Bsq) - D
        Dbar        = mu1 * rho1 * W1 + mu2 * rho2 * W2
        Sbarx       = mu1 * Z1 * vx1 + mu2 * Z2 * vx2
        Sbary       = mu1 * Z1 * vy1 + mu2 * Z2 * vy2
        Sbarz       = mu1 * Z1 * vz1 + mu2 * Z2 * vz2
        tauBar      = mu1 * Z1 / W1 - mu1 * p1 + mu2 * Z2 / W2 - mu2 * p2
        Stildex1    = Z1 * vx1
        Stildex2    = Z2 * vx2
        Stildey1    = Z1 * vy1
        Stildey2    = Z2 * vy2
        Stildez1    = Z1 * vz1
        Stildez2    = Z2 * vz2
        Stildex     = Stildex1 + Stildex2
        Stildey     = Stildey1 + Stildey2
        Stildez     = Stildez1 + Stildez2
        tauTilde1   = Z1 / W1
        tauTilde2   = Z2 / W2
        tauTilde    = tauTilde1 + tauTilde2
        

        aux[:] = h1, W1, e1, vsq1, Z1, vE1, D1, Stildex1, Stildey1, Stildez1, \
                 tauTilde1, h2, W2, e2, vsq2, Z2, vE2, D2, Stildex2, Stildey2, \
                 Stildez2, tauTilde2, \
                 Bsq, Esq, \
                 Jx, Jy, Jz, \
                 Stildex, Stildey, Stildez, \
                 tauTilde, \
                 rhoCh, rhoCh0, \
                 ux, uy, uz, W
        cons[:] = D, Sx, Sy, Sz, tau, Dbar, Sbarx, Sbary, Sbarz, tauBar, Bx, By, Bz, Ex, Ey, Ez, psi, phi
        return cons, aux, alpha





#############################################################################
##########                         SRMHD                         ############        
#############################################################################


class SRMHDClass(object):
    def __init__(self, grid, g=5/3):
        """
        (S)pecial (R)elativistic (M)agneto(h)ydro(d)ynamics
        The special relativistic extension to the magnetohydrodynamic eqiuations.
        Flux is given using flux vector splitting method -
        System is two dimensional with conservative
        vector q = (D, Sx, Sy, Sz, tau, Bx, By, Bz),
        primative variables, prim = (rho, v_x, v_y, v_z, p, Bx, By, Bz)
        auxiliary variables, aux = (h, W, e, c)
        constant Gamma=g,
        h = 1 + e + p/rho = specific enthalpy,
        W**2 = (1 - v**2)**-0.5 = lorentz factor,
        p = rho * e * (g - 1)
        e = specific internal energy
        Parameters
        ----------
        grid: object
            The cell object of the simulation class
            Include the structure of the grid, i.e. total number of cells
        g: float (optional)
            The constant, gamma, featured in the gamma-law equation of state
        """
        assert(type(g) == float or type(g) == int or type(g) == np.ndarray), \
        'g must be float (or similar), type(g) = {}'.format(type(g))
        assert(type(grid) == cells.cells), \
        'Must include the grid structure when initiating model'

        self.consLabels = [r'$D$', r'$S_x$', r'$S_y$', r'$S_z$', r'$\Tau$', r'$B_x$', r'$B_y$', r'$B_z$', r'$\phi$']
        self.primLabels = [r'$\rho$', r'$v_x$', r'$v_y$',r'$v_z$', r'$p$', r'$B_x$', r'$B_y$', r'$B_z$']
        self.auxLabels = [r'$h$', r'$W$', r'$e$', r'$c$']
        self.g = g
        self.Nvars = 9
        self.Nprims = 8
        self.Naux = 9
        self.flux = self.fluxFunc
        self.guessVec = np.ones((2, grid.nx + 2 * grid.Nghosts, grid.ny + 2 * grid.Nghosts))




    def fluxFunc(self, q, simulation, direction):
        """
        Calculates the flux through each cell using the flux splitting method.
        f+ and f- are then reconstructed using weno3 and the end flux is given
        by f = f-l + f+r.
        Parameters
        ----------
        q: array of float
            (Nvars, number_of_cells) Value of the field in repective cells
        simulation: class
            The class containing all the data regarding the simulation
        Returns
        -------
        f: array of float
            (Nvars, number_of_cells) The flux of the fields through the cells
        """
        order = 2   # Hard code order of weno

        prims, aux, alpha = self.getPrimitiveVars(q, simulation)
        D, Sx, Sy, Sz, tau, Bx, By, Bz, phi = q
        rho, vx, vy, vz, p, Bx, By, Bz = prims
        h, W, e, c, b0, bx, by, bz, bsq = aux

        f = np.zeros_like(q)
        
        if direction == 0:
            f[0] = D * vx
            f[1] = Sx * vx + (p + bsq/2) - bx * Bx /W
            f[2] = Sy * vx +             - by * Bx /W
            f[3] = Sz * vx +             - bz * Bx /W
            f[4] = (tau + p + bsq/2) * vx - b0 * Bx / W
            f[5] = phi
            f[6] = By * vx - Bx * vy
            f[7] = Bz * vx - Bx * vz
            f[8] = Bx
            
        else:
            f[0] = D * vy
            f[1] = Sx * vy +             - bx * By /W
            f[2] = Sy * vy + (p + bsq/2) - by * By /W
            f[3] = Sz * vy +             - bz * By /W
            f[4] = (tau + p + bsq/2) * vy - b0 * By / W
            f[5] = Bx * vy - By * vx
            f[6] = phi
            f[7] = Bz * vy - By * vz
            f[8] = By
            
            
        

        # Lax-Friedrichs flux splitting
        fplus = 0.5 * (f + alpha * q)
        fminus = 0.5 * (f - alpha * q)

        fpr = np.zeros_like(q)
        fml = np.zeros_like(q)
        flux = np.zeros_like(q)

        # Reconstruct fluxes
        if direction == 0:
            for j in range(q.shape[2]):
                for i in range(order, q.shape[1]-order):
                    for Nv in range(q.shape[0]):
                        fpr[Nv, i, j] = weno_upwind(fplus[Nv, i-order:i+order-1, j], order)
                        fml[Nv, i, j] = weno_upwind(fminus[Nv, i+order-1:i-order:-1, j], order)
            flux[:,1:-1] = fpr[:,1:-1] + fml[:,1:-1]
        else:
            for i in range(q.shape[1]):		
                for j in range(order, q.shape[2]-order):		
                    for Nv in range(q.shape[0]):		
                         fpr[Nv, i, j] = weno_upwind(fplus[Nv, i, j-order:j+order-1], order)		
                         fml[Nv, i, j] = weno_upwind(fminus[Nv, i, j+order-1:j-order:-1], order)		
            flux[:, :, 1:-1] = fpr[:, :, 1:-1] + fml[:, :, 1:-1]
            
        return flux



    def getPrimitiveVars(self, q, sim):
        """
        Function solves for the primative variables from the conserved, to get
        the entire state of the system via fsolve() root finding method.
        Parameters
        ----------
        q: array of floats
            (Nvars, number_of_cells) The conserved variables of the
            system (D, Sx, Sy, Sz, tau, Bx, By, Bz)
        Returns
        -------
        prims: array of float
            (Nvars, number_of_cells) The primative variables of the
            system (rho, vx, vy, vz, p)
        aux: array of float
            (4, numer_of_cells) The auxiliary variables, including
            h = enthalpy, W = Lorentz factor, e = internal energy,
            c = sound speed
        """
        prims = np.zeros((self.Nprims, q.shape[1], q.shape[2]))
        aux = np.zeros((self.Naux, q.shape[1], q.shape[2]))

        D, Sx, Sy, Sz, Tau, Bx, By, Bz, phi = q
        BS = Bx * Sx + By * Sy + Bz * Sz
        Bsq = Bx**2 + By**2 + Bz**2
        BSsq = BS**2
        Ssq = Sx**2 + Sy**2 + Sz**2

        def residual(guess, i, j):
            vsq, Z = guess
            if vsq >= 1 or Z < 0:
                return 1e6*np.ones_like(guess)
                
            W = 1 / np.sqrt(1 - vsq)
            rho = q[0, i, j] / W
            h = Z / (rho * W**2)
            p = (h - 1) * rho * (self.g - 1) / self.g

            if p < 0 or rho < 0 or h < 0 or W < 1 or np.isnan(W) or np.isnan(rho):
                return 1e6 * np.ones_like(guess)

            Ssqbar = (Z + Bsq[i, j])**2*vsq - (2*Z + Bsq[i, j]) * BSsq[i, j] / Z**2
            taubar = Z + Bsq[i, j] - p - Bsq[i, j] / (2 * W**2) - BSsq[i, j] / (2 * Z**2) - D[i, j]

            residual = np.ones_like(guess)
            residual[0] = Ssqbar - Ssq[i, j]
            residual[1] = taubar - Tau[i, j]
            if np.isnan(residual[0]) or np.isnan(residual[1]):
                return 1e6 * np.ones_like(guess)
            return residual


        # Loop over all cells and solve individually using fsolve, ensuring that p>0
        # guessVec is a (2, Ncells) array, guessVec[1] = Z = rho * h, and
        # guessVec[0] = v**2

            
        for i in range(q.shape[1]):
            for j in range(q.shape[2]):
                vsqGuess =  sim.prims[1, i, j]**2 + sim.prims[2, i, j]**2 + sim.prims[3, i, j]**2
                Wsq = 1 / (1 - vsqGuess)
                Zguess = sim.prims[0, i, j] * sim.aux[0, i, j] * Wsq
                # Solve
                self.guessVec[:, i, j] = fsolve(residual, [vsqGuess, Zguess], args=(i,j))


        vsq, Z = self.guessVec
        W = 1 / np.sqrt(1 - vsq)  
        
        # Rootfinding can give some annoying results, so check result is sensible (vsq < 1)
        # If solution is silly, and the normal method for guessing did not work, use
        # the arithmetic average of the surrounding cells' solutions as an initial guess
        if np.any(np.isnan(W)):
            # Get cell indices where vsq>1
            coords = np.argwhere(np.isnan(W))
            for n in range(coords.shape[0]):
                x = coords[n, 0]
                y = coords[n, 1]
                Nx = sim.cells.nx + 2*sim.cells.Nghosts
                Ny = sim.cells.ny + 2*sim.cells.Nghosts
                comp = []
                # For each of the surrounding cells (not diagonal), ensure it exists,
                # and that it too has solved to a sensible solution. If so, save x/y indices
                for xdir in [-1, 0, 1]:
                    for ydir in [-1, 0, 1]:
                        if ((xdir == 0) != (ydir == 0)) and x+xdir >= 0 and x+xdir < Nx and y+ydir >= 0 and y+ydir < Ny \
                        and self.guessVec[0, x+xdir, y+ydir] < 1 and not np.isnan(self.guessVec[0, x+xdir, y+ydir]):
                            comp.append([x+xdir, y+ydir])
                tot = np.zeros(2)
                # Take average of each of the valid neighbours solutions and use a guess
                for elem in comp:
                    tot += self.guessVec[:, elem[0], elem[1]]
                avgGuess = tot/len(comp)             
                self.guessVec[:, x, y] = fsolve(residual, avgGuess, args=(x, y))
                # Check that this solution makes sense.
                vsq[x, y], Z[x, y] = self.guessVec[:, x, y]
                W[x, y] = 1 / np.sqrt(1-vsq[x, y])
                # If it doesnt, shit is the final word....
                if np.isnan(W[x, y]):
                    print("New guessing didnt work... shit. Guess of {} gave vsq={} for cell ({}, {})".format(avgGuess[0], vsq[x, y], x, y))
                    if x <sim.cells.Nghosts-2 or y<sim.cells.Nghosts-2:
                        # This is in the halo, ignore?
                        print("Error in halo, ignoring")
                    else:
#                        print("comp was:\n{}".format(comp))
#                        for com in comp:
#                            print("{}".format(self.guessVec[0, com[0], com[1]]))
#                        print("len(comp) = {}, tot = {}".format(len(comp), tot))
#                        import sys
#                        sys.exit(1)
                        res = fsolve(residual, avgGuess, args=(x, y))
                        if res[0] < 1:
                            print("WORKS NOW")
                            self.guessVec[:, x, y] = res
                        else:
                            print("comp was:\n{}".format(comp))
                            for com in comp:
                                print("{}".format(self.guessVec[0, com[0], com[1]]))
                            print("len(comp) = {}, tot = {}".format(len(comp), tot))
                            import sys
                            sys.exit(1)
                    
                    
                    
        rho = D / W
        h = Z / (rho * W**2)
        p = (h - 1) * rho * (self.g - 1) / self.g
        e = p / (rho * (self.g - 1))
        vx = (Bx * BS + Sx * Z) / (Z * (Bsq + Z))
        vy = (By * BS + Sy * Z) / (Z * (Bsq + Z))
        vz = (Bz * BS + Sz * Z) / (Z * (Bsq + Z))
        c = np.sqrt((e * self.g * (self.g - 1)) / h)
#        c = 1 # fudge
        b0 = W * (Bx * vx + By * vy + Bz * vz)
        bx = Bx / W + b0 * vx
        by = By / W + b0 * vy
        bz = Bz / W + b0 * vz
        bsq = (Bsq  + b0**2) / W**2

        # Calculate maximum wave speed
        alpha = 1   # Lazines

        prims[:] = rho, vx, vy, vz, p, Bx, By, Bz
        aux[:] = h, W, e, c, b0, bx, by, bz, bsq

        return prims, aux, alpha

    def getConsFromPrims(self, prims):
        """
        Uses the values of the primitive variables to determine the corresponding
        conserved vector. Required before the simulation is evolved due to the
        initial state being given in terms of the prims. Also computes and returns
        the auxilliary variables.
        """
        rho, vx, vy, vz, p, Bx, By, Bz = prims
        vsq = vx**2 + vy**2 + vz**2
        W = 1 / np.sqrt(1 - vsq)
        b0 = W * (Bx * vx + By * vy + Bz * vz)
        bx = Bx / W + b0 * vx
        by = By / W + b0 * vy
        bz = Bz / W + b0 * vz
        bsq = ((Bx**2 + By**2 + Bz**2) + b0**2) / W**2
        h = 1 + p / rho * (self.g / (self.g - 1))
        e = p / (rho * (self.g - 1))
        c = np.sqrt((e * self.g * (self.g - 1)) / h)
        
        cons = np.zeros((self.Nvars, rho.shape[0], rho.shape[1]))
        aux = np.zeros((self.Naux, rho.shape[0], rho.shape[1]))
        
        cons[0] = rho * W
        cons[1] = (rho*h + bsq) * W**2 * vx - b0 * bx
        cons[2] = (rho*h + bsq) * W**2 * vy - b0 * by
        cons[3] = (rho*h + bsq) * W**2 * vz - b0 * bz
        cons[4] = (rho*h + bsq) * W**2 - (p + bsq/2) - b0**2 - cons[0]
        cons[5] = Bx
        cons[6] = By
        cons[7] = Bz
        cons[8] = 0
        aux[:] = h, W, e, c, b0, bx, by, bz, bsq
        # Calculate maximum wave speed
        alpha = 1   # Lazines

        return cons, aux, alpha












#############################################################################
##########                         SRHD                          ############        
#############################################################################

class relEulerGammaLawClass(object):
    def __init__(self, grid=None, g=1.4):
        """
        Contains the system described in Leveque §14.3 - the nonrelativistic
        euler equations. Flux is given exactly by Roe's method (riemannFlux()
        function), or approximately using flux vector splitting (fluxFunc).
        System is one dimensional with three variables, and the conservative
        vector q = (rho, rho*u, E), primative variables prim = (rho, u, p),
        constant gamma=g, and jacobian, A,
        A = [           0            ,             1         ,    0   ]
            [     (g-3)u**2/2        ,           (3-g)u      ,  (g-1) ]
            [   (g-1)*u**3/2-u*H     ,       H-(g-1)*u**2    ,   g*u  ]
        H = (E+p)/rho = total specific enthalpy

        Parameters
        ----------
        grid: object (not needed)
            exists for consistency between euler and relEuler class
        g: float (optional)
            The constant, gamma, featured in the gamma-law equation of state
        """
        assert(type(g) == float or type(g) == int or type(g) == np.ndarray), \
        'g must be float (or similar), type(g) = {}'.format(type(g))

        self.consLabels = [r'$D$', r'$S_x$', r'$S_y$', r'$\tau$']
        self.primLabels = [r'$\rho$', r'$u_x$', r'$u_y$', r'$p$']
        self.auxLabels = [r'$H$', r'$e$', r'$c$']
        self.g = g
        self.Nvars = 4
        self.Nprims = 4
        self.Naux = 3
        self.flux = self.fluxFunc
        self.modA = np.sqrt(self.g)         # |A| for Roe's method




    def fluxFunc(self, q, simulation, direction, order=2):
        """
        Formula of the flux function, using the jacobian of the system

        Parameters
        ----------
        q: array of float
            (Nvars, number_of_cells) Value of the field in repective cells
        simulation: object
            The class containing all the data regarding the simulation
        direction: int
            The direction of the flux, 0=x, 1=y.
        order: float (optional)
            The order of the polynomial reconstruction of the fluxes

        Returns
        -------
        flux: array of float
            (Nvars, number_of_cells) The flux of the fields through the cells
        """
        prims, aux, alpha = self.getPrimitiveVars(q, simulation)

        # Flux splitting method
        # Get flux of each cell
        f = np.zeros_like(q)
        f[0] = q[0] * prims[1]
        if direction == 0:
            f[1] = q[1] * prims[1] + prims[3]
            f[2] = q[2] * prims[1]
        else:
            f[1] = q[1] * prims[1]
            f[2] = q[2] * prims[1] + prims[3]
        f[3] = (q[3] + prims[3]) * prims[1]

        # Lax-Friedrichs flux splitting
        fplus = 0.5 * (f + alpha * q)
        fminus = 0.5 * (f - alpha * q)


        fpr = np.zeros_like(q)
        fml = np.zeros_like(q)
        flux = np.zeros_like(q)

        # Reconstruct fluxes
        if direction == 0:
            for j in range(q.shape[2]):
                for i in range(order, q.shape[1]-order):
                    for Nv in range(q.shape[0]):
                        fpr[Nv, i, j] = weno_upwind(fplus[Nv, i-order:i+order-1, j], order)
                        fml[Nv, i, j] = weno_upwind(fminus[Nv, i+order-1:i-order:-1, j], order)
            flux[:,1:-1] = fpr[:,1:-1] + fml[:,1:-1]
        else:
            for i in range(q.shape[1]):		
                for j in range(order, q.shape[2]-order):		
                    for Nv in range(q.shape[0]):		
                         fpr[Nv, i, j] = weno_upwind(fplus[Nv, i, j-order:j+order-1], order)		
                         fml[Nv, i, j] = weno_upwind(fminus[Nv, i, j+order-1:j-order:-1], order)		
            flux[:, :, 1:-1] = fpr[:, :, 1:-1] + fml[:, :, 1:-1]

        return flux
    

    def getPrimitiveVars(self, q, simulation):
        """
        Function solves for the primative variables from the conserved, to get
        the entire state of the system.
        self.prims = (rho, u, p)

        Parameters
        ----------
        q: array of floats
            (Nvars, number_of_cells) The conserved variables of the
            system (rho, rho*u, E)

        Returns
        -------
        prims: array of float
            (Nvars, number_of_cells) The primative variables of the
            system (rho, u, p)
        aux: array of float
            (3,number_of_cells) The auxiliary variables of the system (H, e, c)
            where H = total enthalpy, e = internal energy, c = sound speed
        """
        prims = np.zeros((self.Nprims, q.shape[1], q.shape[2]))
        alpha = 1
        aux = np.zeros((self.Naux, q.shape[1], q.shape[2]))

        def residual(p, i, j, sign=1):
            D, Sx, Sy, tau = q[:, i, j]
            if p < 0:
                return sign*1e6
            else:
                x = tau + p + D
                vsq = (Sx**2 + Sy**2) / x**2
                W = 1 / np.sqrt(1 - vsq)
                rho = D / W
                h = (tau + p + D) / (D*W)
                e = h - 1 - p/rho
                pstar = rho * e * (self.g - 1)
                
                if vsq >= 1 or W<1 or rho<0 or e<0:
                    return sign*1e6
                
                return p - pstar
            
            
        for i in range(q.shape[1]):
            for j in range(q.shape[2]):
                
                ###################
                ## Newton Method ##
                ###################
                try:
                    p = newton(residual, simulation.prims[3, i, j], args=(i, j))
                except RuntimeError:
                    ###################
                    ## BrentQ method ##
                    ###################
                    if np.allclose(q[1], 0):
                        pmin = 0.1 * (self.g - 1) * q[3, i, j]
                        pmax = 10 * (self.g - 1) * q[3, i, j]
                    else:
                        pmin = 0.01 * max(np.sqrt(q[1, i, j]**2+q[2, i, j]**2) - q[3, i, j] - q[0, i, j] + 1e-10, 0)
                        pmax = 10 * (self.g - 1) * q[3, i, j]
                    
                    try:
                        p = brentq(residual, pmin, pmax, args=(i, j))
                    except ValueError:
                        try:
                            p = brentq(residual, pmin, pmax, args=(i, j, -1))
                        except ValueError:
                            p = pmin + 1e10
                            print("Major CHEATTTT")
                
                D, Sx, Sy, tau = q[:, i, j]
                prims[3, i, j] = p
                x = tau + p + D
                vsq = (Sx**2 + Sy**2) / x**2
                W = 1 / np.sqrt(1 - vsq)
                if np.any(np.isnan(W)):
                    print("NaN encountered, exiting early")
                    simulation.plotPrimHeatmaps()
                    import sys
                    sys.exit(1)
                    
                rho = D / W
                h = (tau + p + D) / (D*W)
                e = h - 1 - p/rho
                ux = Sx / x
                uy = Sy / x
                c = np.sqrt((e * self.g * (self.g - 1)) / h)
                
                prims[:, i, j] = rho, ux, uy, p
                aux[:, i, j] = h, e, c

        return prims, aux, alpha

    def getConsFromPrims(self, prims):
        rho, ux, uy, p = prims
        vsq = ux**2 + uy**2
        W = 1 / np.sqrt(1 - vsq)
        D = rho * W
        e = p / (rho * (self.g - 1))
        h = 1 + e + p/rho
        c = np.sqrt((e * self.g * (self.g - 1)) / h)
        Sx = rho * h * W**2 * ux
        Sy = rho * h * W**2 * uy
        tau = rho * h * W**2 - p - D
        cons = np.zeros((self.Nvars, prims.shape[1], prims.shape[2]))
        cons[:] = D, Sx, Sy, tau
        aux = np.zeros((self.Naux, prims.shape[1], prims.shape[2]))
        aux[:] = h, e, c
        alpha = 1

        return cons, aux, alpha