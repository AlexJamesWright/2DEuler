"""
timeEv.py

script stores the timestep solvers e.g. Euler, RK2 etc.
"""
from scipy.optimize import fsolve
import numpy as np



def euler(simulation, q):
    """
    Parameters
    ----------
    simulation : object
        The class containing all the data regarding the simulation
    q : numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q : numpy array of floats
        The new values of the field according to the flux given by
        simulation.F using the Euler approximation
        First order accurate in time
    """
    dt = simulation.deltaT
    arg = simulation.F(q, simulation)
    return q - dt * arg

def RK2(simulation, q):
    """
    Parameters
    ----------
    simulation : object
        The class containing all the data regarding the simulation
    q : array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q : array of floats
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 2-step approximation
        Second order accurate in time
    """
    dt = simulation.deltaT
    arg1 = simulation.F(q, simulation)
    p1 = q - dt * arg1
    p1 = simulation.bcs(p1, simulation.cells)
    arg2 = simulation.F(p1, simulation)
    return 0.5 * (q + p1 - dt * arg2)

def RK3(simulation, q):
    """
    Parameters
    ----------
    simulation : object
        The class containing all the data regarding the simulation
    q : array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q : array of floats
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 3-step approximation
        Third order accurate in time
    """
    dt = simulation.deltaT
    arg1 = simulation.F(q, simulation)
    p1 = q - dt * arg1
    p1 = simulation.bcs(p1, simulation.cells)
    simulation.p1 = p1
    arg2 = simulation.F(p1, simulation)
    p2 = 0.25 * (3 * q + p1 - dt * arg2)
    p2 = simulation.bcs(p2, simulation.cells)
    arg3 = simulation.F(p2, simulation)

    return (1/3) * (q + 2 * p2 - 2 * dt * arg3)

def eulerSplitRK3(simulation, q):
    """
    Explicit timestep solver for inclusion of source terms

    Parameters
    ----------
    simulation : object
        The class containing all the data regarding the simulation
    q : array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q : array of floats
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 3-step approximation and the
        forward Euler step

    """

    dt = simulation.deltaT
    qstar = RK3(simulation, q)
    primstar, auxstar, alphastar = simulation.model.getPrimitiveVars(qstar, simulation)
    return qstar + dt * simulation.source(qstar, primstar, auxstar, cp=0.1, eta=1/simulation.model.sig)

def eulerSplitRK2(simulation, q):
    """
    Explicit timestep solver for inclusion of source terms

    Parameters
    ----------
    simulation : object
        The class containing all the data regarding the simulation
    q : array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q : array of floats
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 3-step approximation and the
        forward Euler step

    """

    dt = simulation.deltaT
    qstar = RK2(simulation, q)
    primstar, auxstar, alphastar = simulation.model.getPrimitiveVars(qstar, simulation)
    return qstar + dt * simulation.source(qstar, primstar, auxstar, cp=0.1, eta=1/simulation.model.sig)




def backEulerRK2(simulation, q):
    """
    IMEX timestep solver for stiff source terms. Homogeneous equation solved
    explicitly using Runge-Kutta 2nd order accurate method, source term equation
    solved implicitly using backwards Euler
    Parameters
    ----------
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell
    Returns
    -------
    ans: numpy array of floats
        The new values of the field
    """
    Nx, Ny = q[0].shape
    dt = simulation.deltaT
    qstar = RK2(simulation, q)
    primstar, auxstar, alphastar = simulation.model.getPrimitiveVars(qstar, simulation)
    
    def residual(guess, qstr, prmstr, axstr):
        return guess - qstr.ravel() - dt * simulation.source(qstr, prmstr, axstr, cp=1/simulation.model.kappa, eta=1.0/simulation.model.sig).ravel()

    qnext = np.zeros_like(q)
    qInitGuess = qstar + 0.5*dt*simulation.source(qstar, primstar, auxstar, cp=1/simulation.model.kappa, eta=1.0/simulation.model.sig)

    for i in range(Nx):
        for j in range(Ny):
            qnext[:, i, j] = fsolve(residual, qInitGuess[:, i, j], args=(qstar[:, i, j], primstar[:, i, j], auxstar[:, i, j]))


    return qnext



















def SSP2(simulation, q):
    """
    (S)trong-(S)tability-(P)reserving (2,2,2) IMEX scheme.
    Coefficients taken from Pareschi & Russo '04
    Parameters
    ----------
    simulation: object
        The class containing all the data regarding the simulation
    q : array of floats
        The value of the field at the centre of each corresponding cell
    Returns
    -------
    q : array of floats
        The new values of the field
    """
    temp = simulation
    dt = simulation.deltaT
    gamma = 1 - 1 / np.sqrt(2)
    psi = simulation.source
    Nx, Ny = q[0].shape
    q1 = np.zeros_like(q)
    def residualQ1(guess, i, j):
        # Determine primitive variables as a result of the guess
        guessPrim, guessAux, guessAlpha = simulation.model.getPrimitiveVarsSingleCell(guess, simulation, i, j)
        temp.alpha = guessAlpha

        resid = guess - q[:, i, j].reshape(guess.shape) - dt * gamma \
            * psi(guess.reshape(q[:, i, j].shape), guessPrim, guessAux, cp=0.1, eta=1/simulation.model.sig).reshape(guess.shape)
        if np.any(np.isnan(resid)):
            return 1e6 * np.ones_like(guess)
        else:
            return resid

    for i in range(Nx):
        for j in range(Ny):
            q1[:, i, j] = fsolve(residualQ1, q[:, i, j], args=(i, j))

    guessPrim, guessAux, guessAlpha = simulation.model.getPrimitiveVars(q1, simulation)
    temp.prims = guessPrim
    temp.aux = guessAux
    q1 = simulation.bcs(q1, simulation.cells)
    k1 = -simulation.F(q1, temp)
    psi1 = psi(q1, guessPrim, guessAux, cp=0.1, eta=1/simulation.model.sig)

    q2 = np.zeros_like(q1)
    def residualQ2(guess, i, j):
        # Determine primitive variables as a result of the guess
        if np.any(np.isnan(guess)):
            return 1e6 * np.ones_like(guess)
        guessPrim, guessAux, guessAlpha = simulation.model.getPrimitiveVarsSingleCell(guess, simulation, i, j)
        temp.prims[:, i, j] = guessPrim.reshape(temp.prims[:, i, j].shape)
        temp.aux[:, i, j] = guessAux.reshape(temp.aux[:, i, j].shape)
        temp.alpha = guessAlpha
        psiguess = psi(guess.reshape(q[:, i, j].shape), guessPrim, guessAux, cp=0.1, eta=1/simulation.model.sig).reshape(guess.shape)
        resid = guess - q[:, i, j].reshape(guess.shape) - dt * (k1[:, i, j].reshape(guess.shape) + \
                (1 - 2 * gamma) * psi1[:, i, j].reshape(guess.shape) + gamma * \
                psiguess)
        if np.any(np.isnan(resid)):
            return 1e6 * np.ones_like(guess)
        else:
            return resid

    # Generate a better estimate for q2
    qFlux = q1 + dt * psi1
    qSource = np.zeros_like(qFlux)
    def residualSourceOnly(guess, i, j):
        # Determine primitive variables as a result of the guess
        if np.any(np.isnan(guess)):
            return 1e6 * np.ones_like(guess)
        guessPrim, guessAux, guessAlpha = simulation.model.getPrimitiveVarsSingleCell(guess, simulation, i, j)
        temp.prims[:, i, j] = guessPrim.reshape(temp.prims[:, i, j].shape)
        temp.aux[:, i, j] = guessAux.reshape(temp.aux[:, i, j].shape)
        temp.alpha = guessAlpha
        psiguess = psi(guess.reshape(q[:, i, j].shape), guessPrim, guessAux, cp=0.1, eta=1/simulation.model.sig).reshape(guess.shape)
        resid = guess - q[:, i, j].reshape(guess.shape) - dt * (  \
                (1 - 2 * gamma) * psi1[:, i, j].reshape(guess.shape) + gamma * \
                psiguess)
        if np.any(np.isnan(resid)):
            return 1e6 * np.ones_like(guess)
        else:
            return resid

    for i in range(Nx):
        for j in range(Ny):
            qSource[:, i, j] = fsolve(residualSourceOnly, q1[:, i, j], args=(i, j))
    q2estimate = (qFlux + qSource) / 2

    for i in range(Nx):
        for j in range(Ny):
            q2[:, i, j] = fsolve(residualQ2, q2estimate[:, i, j], args=(i, j))

    guessPrim, guessAux, guessAlpha = simulation.model.getPrimitiveVars(q2, simulation)
    temp.prims = guessPrim
    temp.aux = guessAux


    q2 = simulation.bcs(q2, simulation.cells)
    k2 = -simulation.F(q2, temp)
    psi2 = psi(q2, guessPrim, guessAux, cp=0.1, eta=1/simulation.model.sig)

    # Quick check that variables make sense
    assert(not np.all(np.isnan(simulation.aux[1]))), "Time integrator returns NaN"

    return q + 0.5 * dt * (k1 + k2 + psi1 + psi2)