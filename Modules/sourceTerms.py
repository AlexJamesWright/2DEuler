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

    def sourceF(self, q, cp=1):
        return self.sourceType(q, cp)



def noSource(q):
    """
    No source term - used for homogeneous field equations
    """
    return np.zeros_like(q)

def divCleaning(q, cp):
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
    source = np.zeros_like(q)
    source[8] = -q[8] / cp**2
    return source
