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
    Ng = cells.Nghosts
    

    for i in range(Ng):
        for j in range(cells.Ny):
            q[:, i, j] = q[:, Ng, j]
            q[:, cells.nx + Ng + i, j] = q[:, cells.nx + Ng - 1, j]
          
    if (cells.Ny > 1):
        for i in range(cells.Nx):
            for j in range(Ng):
                q[:, i, j] = q[:, i, Ng]
                q[:, i, cells.ny + Ng + j] = q[:, i, cells.ny + Ng - 1]
        
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
    Ng = cells.Nghosts

    for i in range(Ng):
        for j in range(cells.Ny):
            q[:, i, j] = q[:, -2*Ng+i, j]
            q[:, -Ng+i, j] = q[:, Ng+i, j]
           
    if (cells.Ny > 1):
        for i in range(cells.Nx):
            for j in range(Ng):
                q[:, i, j] = q[:, i, -2*Ng+j]
                q[:, i, -Ng+j] = q[:, i, Ng+j]

    return q    

