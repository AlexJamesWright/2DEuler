import numpy as np
from scipy.optimize import newton, brentq, fsolve
from weno import weno_upwind
import warnings
import cells

warnings.filterwarnings('ignore', 'The iteration is not making good progress')









#############################################################################
##########                         SRMHD                         ############        
#############################################################################


class SRMHDClass(object):
    def __init__(self, grid, g=5/3):
        """
        (S)pecial (R)elativistic (M)agneto(h)ydro(d)ynamics
        The special relativistic extension to the magnetohydrodynamic eqiuations.
        Flux is given using flux vector splitting method -
        System is one dimensional with conservative
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

        self.consLabels = [r'$D$', r'$S_x$', r'$S_y$', r'$S_z$', r'$\Tau$', r'$B_x$', r'$B_y$', r'$B_z$']
        self.primLabels = [r'$\rho$', r'$v_x$', r'$v_y$',r'$v_z$', r'$p$', r'$B_x$', r'$B_y$', r'$B_z$']
        self.auxLabels = [r'$h$', r'$W$', r'$e$', r'$c$']
        self.g = g
        self.Nvars = 8
        self.Nprims = 8
        self.Naux = 9
        self.flux = self.fluxFunc
        self.guessVec = np.ones((2, grid.nx + 2 * grid.Nghosts, grid.ny + 2 * grid.Nghosts))
        self.guessVecm1 = np.ones((2, grid.nx + 2 * grid.Nghosts, grid.ny + 2 * grid.Nghosts))




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
        D, Sx, Sy, Sz, tau, Bx, By, Bz = q
        rho, vx, vy, vz, p, Bx, By, Bz = prims
        h, W, e, c, b0, bx, by, bz, bsq = aux

        f = np.zeros_like(q)
        f[0] = D * vx
        if direction == 0:
            f[1] = Sx * vx + (p + bsq/2) - bx * Bx /W
            f[2] = Sy * vx + - by * Bx /W
            f[3] = Sz * vx + - bz * Bx /W
        else:
            f[1] = Sx * vx + - bx * Bx /W
            f[2] = Sy * vx + (p + bsq/2) - by * Bx /W
            f[3] = Sz * vx + - bz * Bx /W
        f[4] = (tau + p + bsq/2) * vx - b0 * Bx / W
        f[6] = By * vx - Bx * vy
        f[7] = Bz * vx - Bx * vz


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

        D, Sx, Sy, Sz, Tau, Bx, By, Bz = q
        BS = Bx * Sx + By * Sy + Bz * Sz
        Bsq = Bx**2 + By**2 + Bz**2
        BSsq = BS**2
        Ssq = Sx**2 + Sy**2 + Sz**2

        def residual(guess, i, j):
            vsq, Z = guess
            if vsq >= 1:
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

        print(sim.iters)

        for i in range(q.shape[1]):
            for j in range(q.shape[2]):
                self.guessVecm1[:, i, j] = self.guessVec[:, i, j]
                if sim.iters < 2:
                    Zguess = sim.prims[0, i, j] * sim.aux[0, i, j]
                    vsqGuess =  sim.prims[1, i, j]**2 + sim.prims[2, i, j]**2 + sim.prims[3, i, j]**2
                    
                    
                self.guessVec[:, i, j] = fsolve(residual, [vsqGuess, Zguess], args=(i,j))


        vsq, Z = self.guessVec
        W = 1 / np.sqrt(1 - vsq)
        rho = D / W
        h = Z / (rho * W**2)
        p = (h - 1) * rho * (self.g - 1) / self.g
        e = p / (rho * (self.g - 1))
        vx = (Bx * BS + Sx * Z) / (Z * (Bsq + Z))
        vy = (By * BS + Sy * Z) / (Z * (Bsq + Z))
        vz = (Bz * BS + Sz * Z) / (Z * (Bsq + Z))
#        c = np.sqrt((e * self.g * (self.g - 1)) / h)
        c = 1 # fudge
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