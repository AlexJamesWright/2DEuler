"""
sourceTerms.py

script contains the functions that return the contribution of the source to
the field equations
"""

import numpy as np


class sources(object):
    def __init__(self, sourceType):
        """
        Parameters
        ----------
        sourceType: callable
            The funcational form of the source term
            Pick one of the functions declared outside the sources class,
            eg. levPsi
        tau: float (optional)
            The timescale at which the source term acts
        beta: float (optional)
            Additional parameter of source term function
        """
        self.sourceType = sourceType
        self.slowSource = 0
        if self.sourceType is divCleaning:
            self.slowSource = 1

    def sourceF(self, q, prims=None, aux=None, cp=None, eta=None):
        return self.sourceType(q, prims, aux, cp, eta)



def noSource(q, prims, aux, cp, eta):
    """
    No source term - used for homogeneous field equations
    """
    return np.zeros_like(q)

def divCleaning(q, prims, aux, cp, eta):
    """
    If doing multi-dimensional MHD, need to ensure divergence free magnetic field
    using div-cleaning - requires source to unphysical phi field. Should be a slow
    source (ie small) so no need to use IMEX schemes.
    Parameters
    ----------
    q: Floats (Nvars, Nx, Ny)
        Conserved vector
    cp: int=1
        Free parameter. Speed any divergence is driven away.
    """
    assert(cp is not None), "Must give source function value for cp"
    source = np.zeros_like(q)
    source[8] = -q[8] / cp**2
    return source

def twoFluidDivClean(q, prims, aux, cp, eta):
    """
    """
    
    assert(cp is not None), "Must give source function value for cp"
    assert(eta is not None), "Must give source function value for eta=1/sigma"
    
    
    
    # Cons
    D, Sx, Sy, Sz, tau, Dbar, Sbarx, Sbary, Sbarz, \
    tauBar, Bx, By, Bz, Ex, Ey, Ez, psi, phi = q
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

    source = np.zeros_like(q)
    
    source[0] = 0
    source[1] = 0
    source[2] = 0
    source[3] = 0
    source[4] = 0
    source[5] = 0
    source[6] = W * Ex + (uy * Bz - uz * By) - eta * (Jx - rhoCh0 * ux)
    source[7] = W * Ey + (uz * Bx - ux * Bz) - eta * (Jy - rhoCh0 * uy)
    source[8] = W * Ez + (ux * By - uy * Bx) - eta * (Jz - rhoCh0 * uz)
    source[9] = ux * Ex + uy * Ey + uz * Ez - eta * (rhoCh - rhoCh0 * W)
    source[10] = 0
    source[11] = 0
    source[12] = 0
    source[13] = -Jx
    source[14] = -Jy
    source[15] = -Jz
    source[16] = rhoCh - psi / cp**2
    source[17] = -phi / cp**2
    
    return source
    
    
    
    
    
    
    
    
    
    
    
    
    
    
