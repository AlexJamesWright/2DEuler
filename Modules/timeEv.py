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
    Explicit timestep solver for inclusion of source terms

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
    primstar, auxstar, alphastar = simulation.model.getPrimitiveVars(qstar, simulation)
    return qstar + dt * simulation.source(qstar, primstar, auxstar, cp=0.1, eta=1/simulation.model.sig)

def eulerSplitRK2(simulation, q):
    """
    Explicit timestep solver for inclusion of source terms

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
    qstar = RK2(simulation, q)
    primstar, auxstar, alphastar = simulation.model.getPrimitiveVars(qstar, simulation)
    return qstar + dt * simulation.source(qstar, primstar, auxstar, cp=0.1, eta=1/simulation.model.sig)
