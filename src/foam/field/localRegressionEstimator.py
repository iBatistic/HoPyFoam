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


import re
import numpy as np
import os
import sys

from src.foam.decorators import timed

class localRegressionEstimator():

    # Shape parameter of the kernel (constant)
    _k = 6

    def __init__(self, volField, mesh):
        self._volField = volField
        self._mesh = mesh

        # Gauss faces points LRE coefficients
        self._cx, self._cy, self._cz = self.makeCoeffs(volField, mesh)

        # Gauss faces points final interpolation coefficients, assembled using cx,cy and cz
        # Each item in list is corresponding face, each face has sublist of Gauss points
        # Each Gauss point has interpolation vector (cx, cy, cz)
        self._c = self.makeCoeffsVector()

    # Return coeffs vector cx for each Gauss points
    def makeCoeffsVector(self):

        coeffs = [[[] for i in range (self._volField._GaussPointsNb)] for x in range(self._mesh.nFaces())]

        for faceI in range(self._mesh.nFaces()):
            for point in range(self._volField._GaussPointsNb):
                for cellN in range(self._volField.Nn):
                    coeffs[faceI][point].append(np.array(\
                        [self._cx[faceI][point][cellN],\
                         self._cy[faceI][point][cellN],\
                         self._cz[faceI][point][cellN]]))
        return coeffs

    def coeffs(self):
        return self._c

    @timed
    def makeCoeffs(self, volField, mesh) -> list[list[np.ndarray]]:

        # File to write data for debug
        fileName = "LRE_coeffs.debug.txt"

        # Remove old file with results
        if (os.path.exists(fileName)):
            os.remove(fileName)

        # Faces interpolation molecules
        molecules = volField._facesInterpolationMolecule

        # Stencils size (this is equal to volField.Nn only at interior cells)
        Nn = [len(stencil) for stencil in molecules]

        # Number of terms in Taylor expression
        Np = volField.Np

        # Coefficient for each face interpolation molecule
        cx = [[] for x in range(mesh.nFaces())]
        cy = [[] for x in range(mesh.nFaces())]
        cz = [[] for x in range(mesh.nFaces())]

        # Face Gauss points
        facesGaussPoints = [sublist[1] for sublist in volField._facesGaussPointsAndWeights]

        # Loop over Gauss points of interior faces
        for faceI in range(mesh.nFaces()):

            # Loop over face Gauss points
            for gpI in facesGaussPoints[faceI]:

                # np.zeros((rows, cols))
                Q = np.zeros((volField.Np, volField.Nn))
                W = np.zeros((volField.Nn, volField.Nn))

                # Loop over neighbours Nn and construct matrix Q, each neighbour cell have its row
                # (n-th column of Q is q(xn-x))
                for i in range(Nn[faceI]):
                    # Neighbour cell centre
                    neiC = mesh.C()[molecules[faceI][i]]

                    # Taylor series terms calculated using nested for loops
                    N = volField.N;
                    pos = int(0);
                    for I in range(N+1):
                        for J in range(N-I+1):
                            fact = np.math.factorial(I) * np.math.factorial(J)
                            Q[pos, i] = pow(neiC[0]-gpI[0], I) * pow(neiC[1]-gpI[1], J) * (1.0/fact)
                            pos += 1;

               # Loop over neighbours Nn and find max distance
                rs = 0.0
                for i in range(Nn[faceI]):
                    neiC = mesh.C()[molecules[faceI][i]]
                    rsNew = max(rs, np.linalg.norm(gpI - neiC))
                    rs = rsNew

                # Loop over neighbours Nn and construct matrix W (diagonal matrix)
                w_diag = np.zeros(volField.Nn)
                for i in range(Nn[faceI]):
                    neiC = mesh.C()[molecules[faceI][i]]
                    dist = np.linalg.norm(neiC - gpI)
                    w_diag[i] = self.weight(abs(dist), rs)

                if(faceI >= mesh.nInternalFaces()):
                    facePatchType = mesh.facePatchType(faceI, self._volField)

                    if(facePatchType == 'empty'):
                        # Save some time by skipping calculations on empty faces
                        # Fill coefficient vectors with zeros to later on avoid having problem
                        # when looping through list
                        cx[faceI].append([0.0 for _ in range(volField.Nn)])
                        cy[faceI].append([0.0 for _ in range(volField.Nn)])
                        cz[faceI].append([0.0 for _ in range(volField.Nn)])
                        continue

                    # Boundary face, add boundary face contribution to Q and W
                    addBoundaryCoeffs = eval("volField.LREcoeffs." + facePatchType)

                    addBoundaryCoeffs(w_diag, Q)

                #Replace diagonal of W with calculated one w_diag
                diagonal_indices = np.diag_indices(W.shape[0])
                W[diagonal_indices] = w_diag

                # Calculate matrix M = Q*W*Q^T
                M = Q @ W @ Q.T

                if (M.shape[0] != M.shape[1] | M.shape[0] == Np):
                    raise ValueError("Matrix M should be Np x Np shape")

                # Calculate matrix A = M^-1*Q*W
                # Depending on condition number choose pinv or inv
                if np.linalg.cond(M) < 1e6:
                    # Matrix is not close to singular, use np.linalg.inv()
                    A = np.linalg.inv(M) @ Q @ W
                else:
                    # Matrix is close to singular, use np.linalg.pinv()
                    A = np.linalg.pinv(M) @ Q @ W

                # Interpolation coefficients for the current Gauss point
                # With above for loop Taylor expression looks like:
                # 1, (y-b), (y-b)^2, (x-a), (x-a)(x-b), (x-a)^2
                cx[faceI].append(list(A[1+N, :]))
                cy[faceI].append(list(A[1, :]))
                cz[faceI].append([0.0 for _ in range(len(list(A[1, :])))])

                # Write matrices to textual file, so they can be checked
                self.writeData(faceI, gpI, Q, W, M, A, fileName)

        return cx, cy, cz

    # Radially symmetric exponential function
    @classmethod
    def weight(self, dist, rs) -> float:
        # Shape parameter of the kernel
        k = self._k

        # Smoothing length
        dm = 2 * rs

        # Weight
        w = (np.exp(-pow(dist / dm, 2) * pow(k, 2)) - np.exp(-pow(k, 2))) \
            / (1 - np.exp(-pow(k, 2)))

        return w

    @classmethod
    def writeData(self, faceI, gpI, Q, W, M, A, fileName):

        np.set_printoptions(precision=5)
        np.set_printoptions(linewidth=np.inf)
        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=sys.maxsize)

        if os.path.exists(fileName):
            file = open(fileName, "a")
        else:
            file = open(fileName, 'w')

        file.write(f'Face number: {faceI} \n')
        file.write(f'Gauss point: {gpI} \n')
        file.write(f"Matrix Q:\n {Q}\n")
        file.write(f"Matrix W:\n {W}\n")
        file.write(f"Matrix M:\n {M}\n")
        file.write(f"Matrix A:\n {A}\n \n")
        file.close()
