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
import sys
from petsc4py import PETSc
from src.foam.decorators import timed

ADD = PETSc.InsertMode.ADD_VALUES

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
        #print(f'Solving system of equations for field {self._psi._fieldName}\n')

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
        print(f'\t{self._psi._fieldName}:    Iterations= {ksp.its}, residual norm = {ksp.norm}')

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

    def __neg__(self):
        '''
        Return a new fvMatrix instance representing the negation of this instance.
        '''
        return fvMatrix(self._psi, -self._source, -self._A)

    def print(self, printPrecision=15, LHS=True, RHS=True):
        self._A.convert('dense')
        T = self._A.getDenseArray()
        if RHS:
            self._source.view()

        if LHS:
            np.set_printoptions(precision=printPrecision, suppress=True, linewidth=np.inf, threshold=np.inf)
            #wr = [T[i:i + 1].tolist() for i in range(0, len(T), 1)]

            #for item in wr:
            #    print(np.array(item))
            print(np.array(T))

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
            diag = self._A.getValues(range(start, end), range(start, end))

            # Invert and multiply with cell volume
            diag = np.linalg.inv(diag) * vol[i]

            diagTensors.append(diag)

        return diagTensors

    def volOverDf(self):

        volOverD = self.volOverD()

        mesh = self._psi._mesh
        psi = self._psi

        # Initialize faces Gauss points values
        volOverDf = np.zeros((mesh.nFaces, self._psi.Ng, 3, 3))

        # Loop over internal faces
        for faceI in range(mesh.nInternalFaces):
            faceStencil = psi._facesInterpolationMolecule[faceI]
            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Gauss point interpolation coefficient vector
                c = psi.LRE().internalCoeffs[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    volOverDf[faceI][i] += volOverD[cellIndex] * c[j]

        # Loop over boundary faces, extrapolate volOverD
        for patch in psi._boundaryConditionsDict:

            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            # Loop over patch faces
            for faceI in range(startFace, startFace + nFaces):

                faceStencil = psi._facesInterpolationMolecule[faceI]
                faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

                # Loop over Gauss points
                for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                    # Gauss point interpolation coefficient vector
                    c = psi.LRE().boundaryCoeffs[faceI-mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil
                    for j, cellIndex in enumerate(faceStencil):
                        volOverDf[faceI][i] += volOverD[cellIndex] * c[j]

        return volOverDf


    def relax(self, relaxation):
        '''
        Implicit matrix under-relaxation
        '''
        mesh = self._psi._mesh
        nCells = mesh.nCells

        nblocks = nCells
        blockSize = self._psi.dim

        # total number of rows or columns
        matSize = nblocks * blockSize

        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes([matSize, matSize])
        A.setType(PETSc.Mat.Type.BAIJ)
        A.setBlockSize(blockSize)

        A.setFromOptions()
        A.setUp()

        for i in range(self._psi._mesh.nCells):
            start = i * blockSize
            end = (i + 1) * blockSize
            # Get values of diagonal matrix for cell i
            diag = self._A.getValues(range(start, end), range(start, end))

            value = ((1.0-relaxation)/relaxation)*diag

            A.setValuesBlocked(i, i, value.flatten(), addv=ADD)

        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        for i in range(self._psi._mesh.nCells):
            start = i * blockSize
            end = (i + 1) * blockSize
            diag = self._A.getValues(range(start, end), range(start, end))

            value = ((1.0-relaxation)/relaxation) * diag @ self._psi._cellValues[i]

            source.setValues(range(i * self._psi.dim, i * self._psi.dim + self._psi.dim), value, ADD)

        # Assemble the matrix and the RHS vector
        A.assemble()
        source.assemble()

        # Add to original matrix
        self._A += A
        self._source += source



