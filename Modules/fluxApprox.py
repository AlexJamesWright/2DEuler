"""
fluxApprox.py

script stores the various approximaions of the flux at a boundary
"""

import numpy as np



def fluxSplitting(q, simulation):
    """
    The flux approximation for the flux vector splitting method

    Parameters
    ----------
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell
    simulation: class
        The class containing all the data regarding the simulation

    Returns
    -------
    netFlux: numpy array of floats
        The net flux through the corresponding cell
        netFlux[i] is the total flux through cell[i]
    """
    fluxX = simulation.model.fluxX(q, simulation)
    fluxY = simulation.model.fluxY(q, simulation)
    netFlux = np.zeros_like(fluxX)
    netFluxX = np.zeros_like(fluxX)
    netFluxY = np.zeros_like(fluxX)
    
    netFluxX[:, 1:-1, :] = (fluxX[:, 2:, :] - fluxX[:, 1:-1, :]) / simulation.deltaX
    netFluxY[:, :, 1:-1] = (fluxY[:, :, 2:] - fluxY[:, :, 1:-1]) / simulation.deltaY  
    netFlux = netFluxX + netFluxY


    return netFlux





