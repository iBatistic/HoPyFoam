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
    def construct(self, psi, operatorName, gamma):

        source, A = self.defineMatrix(psi, operatorName, gamma)

        return self(psi, source, A)

    def solve(self):
        print('Solving system of equations\n')

        # Create Krylov Subspace Solver
        ksp = PETSc.KSP()
        ksp.create(comm=self._A.getComm())

        # Solver is Conjugate Gradient
        ksp.setType(PETSc.KSP.Type.CG)

        # Set preconditioner (PC) to GAMG (Geometric-Algebraic MultiGrid)
        #ksp.getPC().setType(PETSc.PC.Type.GAMG)
        ksp.getPC().setType(PETSc.PC.Type.LU)

        # Add matrix to solver
        ksp.setOperators(self._A)

        # Solver settings
        ksp.max_it = 1000
        ksp.rtol = 1e-7
        ksp.atol = 0

        # Create right vector and solve system of equations
        x = self._A.createVecRight()
        ksp.solve(self._source, x)
        print("\t Iterations= %d, residual norm = %g" % (ksp.its, ksp.norm))

        # Get solution
        sol = x.getArray()
        # Solution is one big array, here it is divided into sublist to create
        # vector or scalar value for each cell
        dim = self._psi._dimensions
        self._psi._cellValues = [sol[i:i + dim].tolist() for i in range(0, len(sol), dim)]

    # Adding two fvMatrices
    def __add__(self, other):
        A = self._A + other._A
        source = other._source + other._source
        return fvMatrix(self._psi, source, A)