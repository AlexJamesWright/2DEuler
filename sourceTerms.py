"""
sourceTerms.py

script contains the functions that return the contribution of the source to
the field equations
"""

import numpy as np


class sources(object):
    def __init__(self, sourceType, tau=0.01, beta=0.8):
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
        self.tau = tau
        self.beta = beta
        self.sourceType = sourceType

    def sourceF(self, q, simulation=None, i=None):
        return self.sourceType(q, self.tau, self.beta, simulation, i)


def resistiveSRMHD(q, tau, beta, simulation, i):
    if i:
        source = np.zeros_like(q)
        source[8:11] = - simulation.aux[4:7, i]
        source[12] = -simulation.model.kappa*q[12]
    else:
        source = np.zeros_like(q)
        source[8:11] = - simulation.aux[4:7]
        source[12] = -simulation.model.kappa*q[12]

    return source


def granular(q, tau, beta, simulation):
    prims, aux, alpha = simulation.model.getPrimitiveVars(q)
    N = simulation.cells.number_of_cells
    Ng = simulation.cells.Nghosts
    half = (N + 2*Ng) // 2
    g = -9.81
    e = simulation.model.e
    source = np.zeros_like(q)
    source[1, :half] = g * prims[0, :half]
    source[1, half:] = -g * prims[0, half:] # Mirror system at NoFluxBoundary - gravity has directionality
    source[2] = (-(1 - e ** 2) * simulation.model.G(prims[0]) * prims[0] ** 2
                  * aux[2] ** (3./2.)) / tau

    return source

def levPsi(q, tau, beta, sim):
    """
    Function returns the source term used in Leveque §17.15 (p401)
    i.e. source(q) = psi(q)
    For small tau (~1e-5), this source term results in a stiff system. Large
    source terms correspond to approximately no source

    Parameters
    ----------
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell
    tau: float (optional)
        The timescale at which the source term acts
    beta: float (optional)
        Additional parameter of source term function (equal to wave speed in
        linear case)

    Returns
    -------
    psi: numpy array of floats
        The contribution of the source due to the field q and each
        corresponding cell
    """

    return  (1. / tau) * q * (1. - q) * (q - beta)


def levRelax(q, tau, beta, sim):
    """
    Function return the source term used in Leveque §17.17, example 17.7
    f(u) = burgers equation = 1/2 * u ** 2

    Paramters
    ---------
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell
    tau: float (optional)
        The timescale at which the source term acts
    beta: float (redundant)
        Not used for levRelax() but included for consistency in source term classes

    Returns
    -------
    psi: numpy array of floats
        The contribution of the source due to the field q and each
        corresponding cell
    """

    f = lambda x : 0.5 * x ** 2
    sourceContrib = np.zeros_like(q)
    sourceContrib[1, :] = (f(q[0, :]) - q[1, :]) / tau
    return sourceContrib


def noSource(q, tau, beta, sim):
    """
    No source term - used for homogeneous field equations
    """
    return np.zeros_like(q)
