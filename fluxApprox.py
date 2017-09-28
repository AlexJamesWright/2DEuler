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
    netFlux = netFluxX = netFluxY = np.zeros_like(fluxX)
    
    for var in range(q.shape[0]):
        for i in range(1, q.shape[1]-1):
            for j in range(q.shape[2]):
                netFluxX[var, i, j] = (fluxX[var, i+1, j] - fluxX[var, i, j]) / simulation.deltaX
    
    for var in range(q.shape[0]):
        for i in range(q.shape[1]):
            for j in range(1, q.shape[2]-1):
                netFluxY[var, i, j] = (fluxY[var, i, j+1] - fluxY[var, i, j]) / simulation.deltaY
    
    
#    netFluxX[:, 1:-1, :] = (fluxX[:, 2:, :] - fluxX[:, 1:-1, :]) / simulation.deltaX
#    netFluxY[:, :, 1:-1] = (fluxY[:, :, 2:] - fluxY[:, :, 1:-1]) / simulation.deltaY    
    netFlux = netFluxX + netFluxY

    print("Xflux:")
    print((fluxX[:, 2:, :] - fluxX[:, 1:-1, :])[1, 51:57, 3:7] / simulation.deltaX)
    print(netFluxX[1, 50:58 ,3:7])

    return netFlux





