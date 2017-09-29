import numpy as np
from scipy.optimize import newton, brentq
from weno import weno_upwind
import warnings

warnings.filterwarnings('ignore', 'The iteration is not making good progress')

class eulerGammaLawClass(object):
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
        self.fluxX = self.fluxFuncX
        self.fluxY = self.fluxFuncY
        self.modA = np.sqrt(self.g)         # |A| for Roe's method




    def fluxFuncX(self, q, simulation, order=2):
        """
        Formula of the flux function, using the jacobian of the system

        Parameters
        ----------
        q: array of float
            (Nvars, number_of_cells) Value of the field in repective cells
        simulation: object
            The class containing all the data regarding the simulation
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
        f[1] = q[1] * prims[1] + prims[3]
        f[2] = q[2] * prims[1]
        f[3] = (q[3] + prims[3]) * prims[1]

        # Lax-Friedrichs flux splitting
        fplus = 0.5 * (f + alpha * q)
        fminus = 0.5 * (f - alpha * q)


        fpr = np.zeros_like(q)
        fml = np.zeros_like(q)
        flux = np.zeros_like(q)

        # Reconstruct fluxes
        for j in range(q.shape[2]):
            for i in range(order, q.shape[1]-order):
                for Nv in range(q.shape[0]):
                    fpr[Nv, i, j] = weno_upwind(fplus[Nv, i-order:i+order-1, j], order)
                    fml[Nv, i, j] = weno_upwind(fminus[Nv, i+order-1:i-order:-1, j], order)
        flux[:,1:-1] = fpr[:,1:-1] + fml[:,1:-1]

        return flux
    
    
    def fluxFuncY(self, q, simulation, order=2):
        """
        Formula of the flux function, using the jacobian of the system

        Parameters
        ----------
        q: array of float
            (Nvars, number_of_cells) Value of the field in repective cells
        simulation: object
            The class containing all the data regarding the simulation
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
        f[0] = q[0] * prims[2]
        f[1] = q[1] * prims[2]
        f[2] = q[2] * prims[2] + prims[3]
        f[3] = (q[3] + prims[3]) * prims[2]

        # Lax-Friedrichs flux splitting
        fplus = 0.5 * (f + alpha * q)
        fminus = 0.5 * (f - alpha * q)

        fpr = np.zeros_like(q)
        fml = np.zeros_like(q)
        flux = np.zeros_like(q)

        # Reconstruct fluxes
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
                
                if vsq > 1 or W<1 or rho<0 or e<0:
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