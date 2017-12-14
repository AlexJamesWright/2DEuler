"""
cells.py

class to store cell geometry information
"""
import numpy as np

class cells(object):

    def __init__(self, nx, ny=1, xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5, Nghosts=4):
        """
        Class containing the information regarding the set up of the grid.

        Parameters
        ----------
        number_of_cells: int
            The number of interior grid cells desired for the simulation
        xmin: float (optional)
            Left hand edge coordinate of the interior grid cells
        xmax: float (optional)
            Right hand edge coordinate of the interior grid cells
        Nghosts: int (optional)
            Number of 'ghost' cells either side of xmin and xmax. Higher order
            weno schemes require more ghost cells to eliminate edge effects
        """

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Lx = xmax - xmin
        self.Ly = ymax - ymin
        self.Nghosts = Nghosts
        self.deltaX = self.Lx / nx
        self.nx = nx
        self.Nx = nx + 2*Nghosts

        if (ny==0):
            self.Ny = 1
            self.ny = 0
            self.Ly = np.inf
            self.deltaY = np.inf
        else:
            self.deltaY = self.Ly / ny
            self.ny = ny
            self.Ny = ny + 2*Nghosts
        


    def coordinates(self):
        """
        Computes the coordinate position of all grid points - includes ghost
        cells

        Returns
        -------
        lst: numpy array of floats
            The central coordinate of all cells, include Nghost ghost cells
            at either end (typically 4 either end)
        """
        x_start = self.xmin - self.deltaX * self.Nghosts + 0.5 * self.deltaX
        x_end = self.xmax + self.deltaX * self.Nghosts - 0.5 * self.deltaX
        y_start = self.ymin - self.deltaY * self.Nghosts + 0.5 * self.deltaY
        y_end = self.ymax + self.deltaY * self.Nghosts - 0.5 * self.deltaY
        lstX = np.linspace(x_start,
                           x_end, self.nx + 2 * self.Nghosts)
        lstY = np.linspace(y_start,
                           y_end, self.ny + 2 * self.Nghosts)
        return lstX, lstY

    def realCoordinates(self):
        """
        Computes the coordinate position of all interior grid points -
        excludes ghost cells
        Returns
        -------
        lst: numpy array of floats
            The central coordinates of the inner cells (i.e. NOT the
            ghost cells)
        """
        lstX = np.linspace(self.xmin + 0.5 * self.deltaX,
                          self.xmax - 0.5 * self.deltaX, self.nx)
        lstY = np.linspace(self.ymin + 0.5 * self.deltaY, 
                           self.ymax - 0.5 * self.deltaY, self.ny)
        return lstX, lstY

    def boundaries(self):
        """
        Computes the left and right cell boundaries of all cells, starting
        from the left-most ghost cell, storing them as follows:
            [cell[0].leftedge, cell[0].rightedge, cell[1].leftedge, ..., ]
        All points that are not at the extreme edge of the system are counted
        twice. Includes ghost cells.

        Returns
        -------
        mixed: numpy array of floats
            The left and right coordinate boundary of every cell, from left to
            right - INCLUDING ghost cells
        """
        lstX, lstY = self.coordinates()
        leftsX = lstX - 0.5 * self.deltaX
        rightsX = lstX + 0.5 * self.deltaX
        mixedX = np.zeros((2 * np.shape(leftsX)[0]))
        mixedX[::2] = leftsX
        mixedX[1::2] = rightsX
        
        leftsY = lstY - 0.5 * self.deltaY
        rightsY = lstY + 0.5 * self.deltaY
        mixedY = np.zeros((2 * np.shape(leftsY)[0]))
        mixedY[::2] = leftsY
        mixedY[1::2] = rightsY
        return mixedX, mixedY

    def realBoundaries(self):
        """
        Computes the left and right cell boundaries of all cells, starting
        from the left-most interior cell, storing them as follows:
            [cell[0].leftedge, cell[0].rightedge, cell[1].leftedge, ..., ]
        All points that are not at the extreme edge of the system are counted
        twice. Excludes ghost cells.

        Returns
        -------
        mixed: numpy array of floats
            The left and right coordinate boundary of the inner cells, from
            left to right - no ghost cells
        """
        lstX, lstY = self.realCoordinates()
        leftsX = lstX - 0.5 * self.deltaX
        rightsX = lstX + 0.5 * self.deltaX
        mixedX = np.zeros((2 * np.shape(leftsX)[0]))
        mixedX[::2] = leftsX
        mixedX[1::2] = rightsX
        
        leftsY = lstY - 0.5 * self.deltaY
        rightsY = lstX + 0.5 * self.deltaY
        mixedY = np.zeros((2 * np.shape(leftsY)[0]))
        mixedY[::2] = leftsY
        mixedY[1::2] = rightsY
        
        return mixedX, mixedY