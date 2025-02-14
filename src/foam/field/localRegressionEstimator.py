"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Local Regression Estimator
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['localRegressionEstimator']

import numpy as np
import os
import sys
import math
import logging
logging.basicConfig(level=logging.WARNING)
import matplotlib.pyplot as plt
from src.foam.decorators import timed

class localRegressionEstimator():

    def __init__(self, volField, mesh, cellLRE=False, k=6, cond_threshold=1e12):
        self._volField = volField
        self._mesh = mesh
        # Shape parameter of the kernel
        # Value of this constant is taken from Castrillo et al. paper
        self._k = k
        self._cond_threshold = cond_threshold

        # Faces Gauss points LRE coefficients
        self._c, self._cx = self.makeIntCoeffs(volField, mesh)

        # Boundary faces Gauss points LRE coefficients
        self._bC, self._bCx = self.makeBouCoeffs(volField, mesh, False)

        # Boundary faces Gauss points LRE coefficients with ghost points
        self._bgC, self._bgCx = self.makeBouCoeffs(volField, mesh, True)

        if cellLRE:
            self._cellCx = self.makeCellGradCoeffs(volField, mesh)

    @property
    def cellGradCoeffs(self):
        return self._cellCx

    @property
    def internalGradCoeffs(self):
        return self._cx

    @property
    def internalCoeffs(self):
        return self._c

    @property
    def boundaryGradCoeffs(self):
        return self._bCx

    @property
    def boundaryCoeffs(self):
        return self._bC

    @property
    def boundaryGradCoeffsGhost(self):
        return self._bgCx

    @property
    def boundaryCoeffsGhost(self):
        return self._bgC

    def makeIntCoeffs(self, volField, mesh):
        """Calculate the interpolation coefficients for internal faces."""
        return self.makeCoeffs(mesh.nInternalFaces)

    def makeBouCoeffs(self, volField, mesh, ghostBoundaryPoint):
        """Calculate the interpolation coefficients for boundary faces."""
        return self.makeCoeffs(mesh.boundaryFaceNb, mesh.nInternalFaces, ghostBoundaryPoint)

    def makeCoeffs(self, nFaces, faceOffset=0, ghostBoundaryPoint=False)-> tuple[list[list[float]], list[list[np.ndarray]]]:
        """
        Generalized function to calculate interpolation coefficients for
        internal or boundary faces.

        Args:
            nFaces: Number of faces to process.
            faceOffset: Offset for face indexing (0 for internal, mesh.nInternalFaces for boundary).
            ghostBoundaryPoint: Boolean indicating if ghost boundary points should be considered.

        Returns:
            Tuple containing:
            - c: List of coefficients.
            - cx: List of gradient coefficients.
        """
        c = [[] for _ in range(nFaces)]
        cx = [[[] for _ in range(self._volField.Ng)] for _ in range(nFaces)]

        for faceI in range(nFaces):
            faceIdx = faceI + faceOffset
            c_, cx_, cy_, cz_ = self.makeFaceCoeffs(faceIdx, ghostBoundaryPoint)
            c[faceI].extend(c_)

            for point in range(self._volField.Ng):
                for cellN in range(self._volField.Nn):
                    cx[faceI][point].append(\
                        np.array([cx_[point][cellN],\
                                  cy_[point][cellN],\
                                  cz_[point][cellN]]))

        return c, cx

    def makeFaceCoeffs(self, faceI, ghostBoundaryPoint=False):

        c = []
        cx = []
        cy = []
        cz = []

        volField = self._volField
        mesh = self._mesh

        # Face molecule
        molecule = volField._facesInterpolationMolecule[faceI]

        # Molecule size (this is equal to volField.Nn only at interior cells)
        Nn=len(molecule)

        # Number of terms in Taylor expression
        Np = volField.Np

        # Taylor polynomial order
        N = volField.N

        faceGaussPoints = [_[1] for _ in volField._facesGaussPointsAndWeights][faceI]

        # Loop over face neighbours Nn and find max distance
        rs = 0.0
        for i in range(Nn):
            neiC = mesh.C[molecule[i]]
            rsNew = max(rs, np.linalg.norm(mesh.Cf[faceI] - neiC))
            rs = rsNew

        # Loop over face Gauss points
        for gpI in faceGaussPoints:
            # np.zeros((rows, cols))
            Q = np.zeros((volField.Np, volField.Nn))
            W = np.zeros((volField.Nn, volField.Nn))

            # Loop over neighbours Nn and construct matrix Q, each neighbour
            # cell have its row (n-th column of Q is q(xn-x))
            for i in range(Nn):
                # Neighbour cell centre
                neiC = mesh.C[molecule[i]]

                # Taylor series terms calculated using nested for loops
                pos = int(0)
                for I in range(N + 1):
                    for J in range(N - I + 1):
                        fact = math.factorial(I) * math.factorial(J)
                        Q[pos, i] = (pow(neiC[0] - gpI[0], I) * pow(neiC[1] - gpI[1], J) * (1.0 / fact))
                        pos += 1;

            # Loop over neighbours Nn and construct matrix W (diagonal matrix)
            w_diag = np.zeros(volField.Nn)
            for i in range(Nn):
                neiC = mesh.C[molecule[i]]
                dist = np.linalg.norm(neiC - gpI)
                w_diag[i] = self.weight(abs(dist), rs)

            # Skip empty faces
            if faceI >= mesh.nInternalFaces:
                if mesh.facePatchType(faceI, volField) == 'empty':
                    c.append([0.0 for _ in range(volField.Nn)])
                    cx.append([0.0 for _ in range(volField.Nn)])
                    cy.append([0.0 for _ in range(volField.Nn)])
                    cz.append([0.0 for _ in range(volField.Nn)])
                    continue

                if ghostBoundaryPoint:
                    # Add ghost point at boundary Gauss point location
                    w_diag[-1] = 1.0
                    Q[0][-1] = 1.0

            # Replace diagonal of W with calculated one w_diag
            diagonal_indices = np.diag_indices(W.shape[0])
            W[diagonal_indices] = w_diag

            Qhat = Q @ np.diag(np.sqrt(np.diag(W)))

            # Perform QR decomposition
            O, R = np.linalg.qr(Qhat.T, mode='reduced')

            Bhat = np.diag(np.sqrt(np.diag(W))) @ np.eye(len(w_diag))

            Rbar = R[:Np, :Np]
            Qbar = O[:, :Np]

            # In numpy there is no canonical way to compute the inverse
            # of an upper triangular matrix.
            A = np.linalg.solve(Rbar, Qbar.T @ Bhat)

            condNumber = np.linalg.cond(Rbar)
            if condNumber > self._cond_threshold:
                logging.warning(f"High condition number (> {self._cond_threshold}) detected for face {faceI}.")

            c.append(list(A[0, :]))
            cx.append(list(A[1 + N, :]))
            cy.append(list(A[1, :]))
            cz.append([0.0 for _ in range(len(list(A[1, :])))])

        return c, cx, cy, cz

    def makeCellGradCoeffs(self, volField, mesh):

        # Cell interpolation molecules
        molecules = volField._cellsInterpolationMolecule

        # Stencils size (this is equal to volField.Nn only at interior cells)
        Nn = [len(molecule) for molecule in molecules]

        # Number of terms in Taylor expression
        Np = volField.Np

        # Taylor series terms calculated using nested for loops
        N = volField.N

        # Coefficient for each cell interpolation molecule
        c = [[] for x in range(mesh.nCells)]
        cx = [[] for x in range(mesh.nCells)]
        cy = [[] for x in range(mesh.nCells)]
        cz = [[] for x in range(mesh.nCells)]

        # Loop over Gauss points of interior faces
        for cellI in range(mesh.nCells):

            # Loop over neighbours Nn and find max distance
            rs = 0.0
            for i in range(Nn[cellI]):
                neiC = mesh.C[molecules[cellI][i]]
                rsNew = max(rs, np.linalg.norm(mesh.C[cellI] - neiC))
                rs = rsNew

            # np.zeros((rows, cols))
            Q = np.zeros((volField.Np, volField.Nn))
            W = np.zeros((volField.Nn, volField.Nn))

            # Loop over neighbours Nn and construct matrix Q, each neighbour cell have its row
            # (n-th column of Q is q(xn-x))
            for i in range(Nn[cellI]):
                # Neighbour cell centre
                neiC = mesh.C[molecules[cellI][i]]

                pos = int(0)
                for I in range(N + 1):
                    for J in range(N - I + 1):
                        fact = math.factorial(I) * math.factorial(J)
                        Q[pos, i] = pow(neiC[0] - mesh.C[cellI][0], I) * pow(neiC[1] - mesh.C[cellI][1], J) * (1.0 / fact)
                        pos += 1;

            # Loop over neighbours Nn and construct matrix W (diagonal matrix)
            w_diag = np.zeros(volField.Nn)
            for i in range(Nn[cellI]):
                neiC = mesh.C[molecules[cellI][i]]
                dist = np.linalg.norm(neiC - mesh.C[cellI])
                w_diag[i] = self.weight(abs(dist), rs)

            # Add boundary face contribution to Q and W
            if cellI in mesh.boundaryCells:

                cellFaces = mesh.cellFaces[cellI]
                for faceI in cellFaces:
                    facePatchType = mesh.facePatchType(faceI, self._volField, False)

                    if Nn[cellI] == volField.Nn - 1:
                        if facePatchType != 'empty' and facePatchType is not None:

                            neiC = mesh.Cf[faceI]

                            pos = int(0)
                            for I in range(N + 1):
                                for J in range(N - I + 1):
                                    fact = math.factorial(I) * math.factorial(J)
                                    Q[pos, -1] = pow(neiC[0] - mesh.C[cellI][0], I) * pow(neiC[1] - mesh.C[cellI][1], J) * (1.0 / fact)
                                    pos += 1;

                            dist = np.linalg.norm(neiC - mesh.C[cellI])
                            w_diag[-1] = self.weight(abs(dist), rs)

                    if Nn[cellI] == volField.Nn - 2:

                        if facePatchType == "pressureTraction":

                            neiC = mesh.Cf[faceI]
                            pos = int(0)
                            for I in range(N + 1):
                                for J in range(N - I + 1):
                                    fact = math.factorial(I) * math.factorial(J)
                                    Q[pos, -1] = pow(neiC[0] - mesh.C[cellI][0], I) * pow(neiC[1] - mesh.C[cellI][1],J) * (1.0 / fact)
                                    pos += 1;

                            dist = np.linalg.norm(neiC - mesh.C[cellI])
                            w_diag[-1] = self.weight(abs(dist), rs)

                        if facePatchType == "zeroGradient":

                            neiC = mesh.Cf[faceI]
                            pos = int(0)
                            for I in range(N + 1):
                                for J in range(N - I + 1):
                                    fact = math.factorial(I) * math.factorial(J)
                                    Q[pos, -2] = pow(neiC[0] - mesh.C[cellI][0], I) * pow(neiC[1] - mesh.C[cellI][1], J) * (1.0 / fact)
                                    pos += 1;

                            dist = np.linalg.norm(neiC - mesh.C[cellI])
                            w_diag[-2] = self.weight(abs(dist), rs)

            # Replace diagonal of W with calculated one w_diag
            diagonal_indices = np.diag_indices(W.shape[0])
            W[diagonal_indices] = w_diag

            Qhat = Q @ np.diag(np.sqrt(np.diag(W)))

            # Perform QR decomposition
            O, R = np.linalg.qr(Qhat.T, mode='reduced')

            Bhat = np.diag(np.sqrt(np.diag(W))) @ np.eye(len(w_diag))

            Rbar = R[:Np, :Np]
            Qbar = O[:, :Np]

            # In numpy there is no canonical way to compute the inverse
            # of an upper triangular matrix.
            A = np.linalg.solve(Rbar, Qbar.T @ Bhat)

            condNumber = np.linalg.cond(Rbar)
            if condNumber > self._cond_threshold:
                logging.warning(f"High condition number (> {self._cond_threshold}) detected for face {faceI}.")

            # Interpolation coefficients for the current Gauss point
            # With above for loop Taylor expression looks like:
            # 1, (y-b), (y-b)^2, (x-a), (x-a)(x-b), (x-a)^2
            c[cellI].extend(list(A[0, :]))
            cx[cellI].extend(list(A[1 + N, :]))
            cy[cellI].extend(list(A[1, :]))
            cz[cellI].extend([0.0 for _ in range(len(list(A[1, :])))])
        #
        # print(len(cx))
        # print(len(cy))
        # print(len(cz))
        # print(cx)
        # for i in range(len(cx)):
        #     print(len(cx[i]))
        #
        # for i in range(len(cy)):
        #     print(len(cy[i]))
        #
        # for i in range(len(cz)):
        #         print(len(cz[i]))

        coeffs = [[] for _ in range(mesh.nCells)]

        for cellI in range(mesh.nCells):
            for cellN in range(volField.Nn):
                #print("CellI", cellI)
                #print("CellN", cellN)
                #print(print([cx[cellI][cellN], cy[cellI][cellN], cz[cellI][cellN]]))
                coeffs[cellI].append([cx[cellI][cellN], cy[cellI][cellN], cz[cellI][cellN]])

        #print(coeffs[90], coeffs[99])
        return coeffs


    def weight(self, dist, rs) -> float:
        """Radially symmetric exponential function for weight calculation"""

        # Shape parameter of the kernel
        k = self._k

        # Smoothing length
        dm = 2 * rs

        # Weight
        w = (np.exp(-pow(dist / dm, 2) * pow(k, 2)) - np.exp(-pow(k, 2))) \
            / (1 - np.exp(-pow(k, 2)))

        return w
