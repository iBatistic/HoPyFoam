"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['bodyForce']


import os
from src.foam.foamFileParser import *
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class bodyForce():

    @classmethod
    def construct(self, psi, *args):

        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

        nblocks = nCells
        blockSize = psi.dim
        matSize = nblocks * blockSize

        # Create empty left left-hand side matrix
        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes([matSize, matSize])
        A.setType(PETSc.Mat.Type.BAIJ)
        A.setBlockSize(blockSize)
        A.setFromOptions()
        A.setUp()

        # Create solution vector
        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # If exists read body force from file
        fileName = 'bodyForce.dat'

        # Components of CV body force are space separated and
        # each cv force has corresponding row
        if os.path.exists(fileName):
            with open(fileName, 'r') as file:
                cellI = 0
                for line in file:
                    parts = line.split()
                    # Body force components
                    bx = float(parts[0])
                    by = float(parts[1])
                    bz = float(parts[2])
                    # Cell volume
                    V = mesh.V[cellI]

                    bodyForce = np.array([bx, by, bz])

                    # Add body force to source vector
                    source.setValues(range(cellI * psi.dim, cellI * psi.dim + psi.dim), bodyForce*V, ADD)
                    cellI += 1

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A