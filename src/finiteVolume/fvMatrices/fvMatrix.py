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
from src.foam.decorators import timed

class fvMatrix():

    def __init__(self, psi, source, A):
        self._A = A
        self._source = source
        self._psi = psi

    @classmethod
    def construct(self, psi, operatorName, gamma=None, cellPsi=None):

        source, A = self.defineMatrix(psi, operatorName, gamma, cellPsi)

        return self(psi, source, A)

    @timed
    def solve(self):
        print(f'Solving system of equations for field {self._psi._fieldName}\n')

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
        ksp.rtol = 1e-8
        ksp.atol = 0

        # Create right vector and solve system of equations
        x = self._A.createVecRight()
        ksp.solve(self._source, x)
        print("\t Iterations= %d, residual norm = %g \n" % (ksp.its, ksp.norm))

        # Get solution
        sol = x.getArray()
        # Solution is one big array, here it is divided into sublist to create
        # vector or scalar value for each cell
        dim = self._psi.dim
        self._psi._cellValues = [sol[i:i + dim].tolist() for i in range(0, len(sol), dim)]

        # Evaluate boundary values
        self._psi.evaluateBoundary()

    def __add__(self, other):
        '''
        Adding two fvMatrices
        '''
        A = self._A + other._A
        source = self._source + other._source
        return fvMatrix(self._psi, source, A)

    def __sub__(self, other):
        '''
        Subtracting two fvMatrices
        '''
        A = self._A - other._A
        source = self._source - other._source
        return fvMatrix(self._psi, source, A)

    def print(self, printPrecision=15):
        self._A.convert('dense')
        T = self._A.getDenseArray()
        self._source.view()
        np.set_printoptions(precision=printPrecision, suppress=True)
        wr = [T[i:i + 1].tolist() for i in range(0, len(T), 1)]
        for item in wr:
            print(np.array(item))

    def volOverD(self):
        '''
        Returns matrix inverse of diagonal coefficients multiplied with cell
        volume. Used for SIMPLE algorithm.
        '''
        blockSize = self._psi.dim
        vol = self._psi._mesh.V
        diagTensors = []
        for i in range(self._psi._mesh.nCells):
            start = i * blockSize
            end = (i + 1) * blockSize
            # Get values of diagonal matrix for cell i
            diag_block = self._A.getValues(range(start, end), range(start, end))

            # Invert and multiply with cell volume
            diag_block = np.linalg.inv(diag_block) * vol[i]

            diagTensors.append(diag_block)

        return diagTensors
