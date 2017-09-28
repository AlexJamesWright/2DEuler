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
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
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
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
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
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
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
    EX timestep solver for inclusion of source terms

    Parameters
    ----------
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 3-step approximation and the
        forward Euler step

    """

    dt = simulation.deltaT
    qstar = RK3(simulation, q)
    return qstar + dt * simulation.source(qstar, simulation)

def backEulerRK3(simulation, q):
    """
    IMEX timestep solver for stiff source terms. Homogeneous equation solved
    explicitly using Runge-Kutta 3rd order accurate method, source term equation
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
        The new values of the field according to the flux given by
        simulation.F using the Runge-Kutta 3-step approximation and the
        backward Euler step
    """
    dt = simulation.deltaT
    qstar = RK3(simulation, q)
    def residual(guess):
        return guess - qstar.ravel() - dt * simulation.source(guess.reshape(q.shape), simulation).ravel()

    initGuess = qstar + 0.5 * dt * simulation.source(qstar, simulation)
    ans = fsolve(residual, initGuess.ravel()).reshape(q.shape)
    return ans


def SSP2(simulation, q):
    """
    (S)trong-(S)tability-(P)reserving (2,2,2) IMEX scheme.
    Coefficients taken from Pareschi & Russo '04

    Parameters
    ----------
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
        The new values of the field
    """
    dt = simulation.deltaT
    gamma = 1 - 1 / np.sqrt(2)
    psi = simulation.source
    Ntot = q.shape[1]
    q1 = np.zeros_like(q)
    def residualQ1(guess, i):
        resid = guess - q[:, i].reshape(guess.shape) - dt * gamma \
        * psi(guess.reshape(q[:, i].shape), simulation, i).reshape(guess.shape)
        return resid

    for i in range(Ntot):
        q1[:, i] = fsolve(residualQ1, q[:, i], args=[i])


    q1 = simulation.bcs(q1, simulation.cells)
    k1 = -simulation.F(q1, simulation)
    psi1 = psi(q1, simulation)

    q2 = np.zeros_like(q1)
    def residualQ2(guess, i):
        resid = guess - q[:, i].reshape(guess.shape) - dt * (k1[:, i].reshape(guess.shape) + \
                (1 - 2 * gamma) * psi1[:, i].reshape(guess.shape) + gamma * \
                psi(guess.reshape(q[:, i].shape), simulation, i).reshape(guess.shape))
        return resid


    for i in range(Ntot):
        q2[:, i] = fsolve(residualQ2, q1[:, i], args=[i])
    q2 = simulation.bcs(q2, simulation.cells)
    k2 = -simulation.F(q2, simulation)
    psi2 = psi(q2, simulation)

    return q + 0.5 * dt * (k1 + k2 + psi1 + psi2)


def SSP3(simulation, q):
    """
    (S)trong-(S)tability-(P)reserving (4,3,3) IMEX scheme.
    Coefficients taken from Pareschi & Russo 2004

    Parameters
    ----------
    simulation: class
        The class containing all the data regarding the simulation
    q: numpy array of floats
        The value of the field at the centre of each corresponding cell

    Returns
    -------
    q: numpy array of floats
        The new values of the field
    """
    dt = simulation.deltaT
    alpha = 0.24169426078821
    beta = 0.06042356519705
    eta = 0.12915286960590
    psi = simulation.source

    def residualQ1(guess):
        return guess - q.ravel() - dt * alpha * psi(guess.reshape(q.shape), simulation).ravel()
    q1 = fsolve(residualQ1, q.ravel()).reshape(q.shape)
    q1 = simulation.bcs(q1, simulation.cells)
    psi1 = psi(q1, simulation)

    def residualQ2(guess):
        return guess - q.ravel() - dt * alpha * (psi(guess.reshape(q.shape), simulation).ravel() - psi1.ravel())
    q2 = fsolve(residualQ2, q1.copy().ravel()).reshape(q.shape)
    q2 = simulation.bcs(q2, simulation.cells)
    k2 = -simulation.F(q2, simulation)
    psi2 = psi(q2, simulation)

    def residualQ3(guess):
        return guess - q.ravel() - dt * k2.ravel() - dt * ((1 - alpha) * psi2.ravel() + alpha * psi(guess.reshape(q.shape), simulation).ravel())
    q3 = fsolve(residualQ3, q2.copy().ravel()).reshape(q.shape)
    q3 = simulation.bcs(q3, simulation.cells)
    k3 = -simulation.F(q3, simulation)
    psi3 = psi(q3, simulation)

    def residualQ4(guess):
        fac = 0.5 - beta - eta - alpha
        arg1 = dt * (k2 + k3) / 4
        arg2 = dt * (beta * psi1 + eta * psi2 + fac * psi3 + alpha * psi(guess.reshape(q.shape), simulation))
        return guess - q.ravel() - arg1.ravel() - arg2.ravel()
    q4 = fsolve(residualQ4, q3.copy().ravel()).reshape(q.shape)
    q4 = simulation.bcs(q4, simulation.cells)
    k4 = -simulation.F(q4, simulation)
    psi4 = psi(q4, simulation)

    return q + dt * (k2 + k3 + 4 * k4 + psi2 + psi3 + 4 * psi4) / 6