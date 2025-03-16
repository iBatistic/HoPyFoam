"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.unizg.hr, philip.cardiff@ucd.ie'
__all__ = ['bodyForce']


import os
from typing import Tuple
import numpy as np
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class bodyForce():

    BODY_FORCE_FILE = "bodyForce.dat"

    @classmethod
    def construct(self, psi, *args):

        if psi.dim == 1:
            return self.scalarBodyForce(psi, *args)
        elif psi.dim == 3:
            return self.vectorBodyForce(psi, *args)
        else:
            raise ValueError("Not supported dimension of psi in bodyForce")

    @classmethod
    def vectorBodyForce(self, psi, *args) -> Tuple[PETSc.Vec, PETSc.Mat]:

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

        # Components of CV body force are  space separated and
        # each cv force has corresponding row
        bodyForce = readBodyForce(self.BODY_FORCE_FILE, psi.dim)

        if bodyForce is not None:
            for cellI in range(mesh.nCells):
                # Cell volume
                V = mesh.V[cellI]

                # Add body force to source vector
                source.setValues(range(cellI * psi.dim, cellI * psi.dim + psi.dim),
                                 bodyForce[cellI] * V, ADD)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A

    @classmethod
    def scalarBodyForce(self, psi, *args) -> Tuple[PETSc.Vec, PETSc.Mat]:

        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

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

        # Components of CV body force are  space separated and
        # each cv force has corresponding row
        bodyForce = readBodyForce(self.BODY_FORCE_FILE, psi.dim)

        if bodyForce is not None:
            for cellI in range(mesh.nCells):
                # Cell volume
                V = mesh.V[cellI]

                # Add body force to source vector
                source.setValues(cellI, bodyForce[cellI] * V, ADD)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A

def readBodyForce(file_path: str, dim: int) -> np.ndarray:
    if not os.path.exists(file_path):
        return None
    body_forces = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < dim:
                raise ValueError(f"Invalid format in {file_path}, "
                                 f"expected {dim} values per line.")

            body_forces.append([float(x) for x in parts[:dim]])
    return np.array(body_forces)