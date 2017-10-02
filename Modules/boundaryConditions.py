"""
boundaryConditions.py

script contains functions to update data in end cells (ghosts) according to
the deired boundary conditions
"""


def outflow(q, cells):
    """
    Imposing flows that exit the system, analogous to a domain that extends to
    infinity in both directions.

    Parameters
    ----------
    q: array of float
        (Nvars, number_of_cells) The value of the convserved variables
        at the centre of each corresponding cell
    cells: object
        A cell object, simulation.cells, containing the information about
        the system's grid construction

    Returns
    -------
    q: array of float
        (Nvars, number_of_cells) The value of the convserved variables
        at the centre of each corresponding cell
    """
    N = cells.Nghosts
    

    for i in range(N):
        for j in range(N, cells.ny + N):
            q[:, i, j] = q[:, N, j]
            q[:, cells.nx + N + i, j] = q[:, cells.nx + N - 1, j]
          
  
    for i in range(N, cells.nx + N):
        for j in range(N):
            q[:, i, j] = q[:, i, N]
            q[:, i, cells.ny + N + j] = q[:, i, cells.ny + N - 1]
        
    return q


def periodic(q, cells):
    """
    Imposing flows that exit the system and re-enter in the other side.
    What flows out of q_N enters at q_0 and viceversa.
    Parameters
    ----------
    q: array of float
        (Nvars, number_of_cells) The value of the convserved variables
        at the centre of each corresponding cell
    cells: object
        A cell object, simulation.cells, containing the information about
        the system's grid construction
    Returns
    -------
    q: array of float
        (Nvars, number_of_cells) The value of the convserved variables
        at the centre of each corresponding cell
    """
    N = cells.Nghosts

    for i in range(N):
        for j in range(N, cells.ny + N):
            q[:, i, j] = q[:, -2*N+i, j]
            q[:, -N+i, j] = q[:, N+i, j]
            
    for i in range(N, cells.nx + N):
        for j in range(N):
            q[:, i, j] = q[:, i, -2*N+j]
            q[:, i, -N+j] = q[:, i, N+j]

    return q    

