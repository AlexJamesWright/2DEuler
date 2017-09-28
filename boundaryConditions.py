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

