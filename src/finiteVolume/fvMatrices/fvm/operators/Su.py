"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.unizg.hr, philip.cardiff@ucd.ie'
__all__ = ['Su']


import os
from typing import Tuple
import numpy as np
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class Su():

    @classmethod
    def construct(self, psi, *args):

        if psi.dim == 1:
            return self.scalarSource(psi, *args)
        elif psi.dim == 3:
            return self.vectorSource(psi, *args)
        else:
            raise ValueError("Not supported dimension of psi in bodyForce")

    @classmethod
    def vectorSource(self, psi, *args) -> Tuple[PETSc.Vec, PETSc.Mat]:

        mesh = psi._mesh
        nCells = mesh.nCells

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

        for cellI in range(mesh.nCells):
            V = mesh.V[cellI]

            source.setValues(range(cellI * psi.dim, cellI * psi.dim + psi.dim), psi._cellValues[cellI] * V, ADD)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A

    @classmethod
    def scalarSource(self, psi, *args) -> Tuple[PETSc.Vec, PETSc.Mat]:

        mesh = psi._mesh
        nCells = mesh.nCells

        # Create empty left left-hand side matrix
        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes((nCells, nCells))
        A.setType(PETSc.Mat.Type.AIJ)
        A.setUp()

        # Create solution vector
        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        for cellI in range(mesh.nCells):
            V = mesh.V[cellI]
            source.setValues(cellI, np.array(psi._cellValues[cellI]) * V, ADD)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A