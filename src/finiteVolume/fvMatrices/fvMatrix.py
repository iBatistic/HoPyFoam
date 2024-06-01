"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description

"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'

import numpy as np
from petsc4py import PETSc

class fvMatrix():

    def __init__(self, psi, source, A):
        self._A = A
        self._source = source
        self._psi = psi

    @classmethod
    def construct(self, psi, operator, gamma):

        source, A = self.defineMatrix(psi, operator, gamma)

        return self(psi, source, A)

    def solve(self):
        print('Solving system of equations\n')

        # Create Krylov Subspace Solver
        ksp = PETSc.KSP()
        ksp.create(comm=self._A.getComm())

        # Solver is Conjugate Gradient
        ksp.setType(PETSc.KSP.Type.CG)

        # Set preconditioner (PC) to GAMG (Geometric-Algebraic MultiGrid)
        ksp.getPC().setType(PETSc.PC.Type.GAMG)

        ksp.setOperators(self._A)

        # Solver settings
        ksp.max_it = 100
        ksp.rtol = 1e-6
        ksp.atol = 0

        # Create right vector and solve system of equations
        x = self._A.createVecRight()
        ksp.solve(self._source, x)
        self._psi._cellValues = x.getArray()

        print("\t Iterations= %d, residual norm = %g" % (ksp.its, ksp.norm))
