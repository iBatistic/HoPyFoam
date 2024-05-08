"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Local Regression Estimator
"""

import re
import numpy as np
import os

from src.foam.decorators import timed

class localRegressionEstimator():

    # Shape parameter of the kernel (constant)
    _k = 6

    _coeffs = "None"

    def __init__(self, volField, mesh):
        self._volField = volField
        self._mesh = mesh

        # Gauss faces points LRE coefficients
        self._cx, self._cy, self._cz = self.makeCoeffs(volField, mesh)

    # Return coeffs vector cx for each Gauss points
    def coeffs(self):

        coeffs = [[[] for i in range (self._volField._GaussPointsNb)] for x in range(self._mesh.nFaces())]

        for faceI in range(self._mesh.nFaces()):
            for point in range(self._volField._GaussPointsNb):
                for cellN in range(self._volField._Nn):
                    coeffs[faceI][point].append(np.array(\
                        [self._cx[faceI][point][cellN],\
                         self._cy[faceI][point][cellN],\
                         self._cz[faceI][point][cellN]]))
        return coeffs

    @timed
    def makeCoeffs(self, volField, mesh) -> list[list[np.ndarray]]:

        # File to write data for debug
        fileName = "LRE_coeffs.txt"

        # Remove old file with results
        if (os.path.exists(fileName)):
            os.remove(fileName)

        # Faces interpolation molecules
        molecules = volField._facesInterpolationMolecule

        # Stencils size (this is equal to volField._Nn only at interior cells)
        Nn = [len(stencil) for stencil in molecules]

        # Number of terms in Taylor expression
        Np = volField._Np

        # Coefficient for each face interpolation molecule
        cx = [[] for x in range(mesh.nFaces())]
        cy = [[] for x in range(mesh.nFaces())]
        cz = [[] for x in range(mesh.nFaces())]

        # Face Gauss points
        facesGaussPoints = [sublist[1] for sublist in volField._facesGaussPointsAndWeights]

        # Loop over Gauss points of interior faces
        for faceI in range(mesh.nFaces()):
            # A quick check for stencil size requirement
            if (Np > Nn[faceI]):
                print(__file__)
            #    raise ValueError(f'Number of neighbours {Nn} is smaller than length of '
            #                     f'Taylor expansion {Np}')

            # Loop over face Gauss points
            for gpI in facesGaussPoints[faceI]:

                # np.zeros((rows, cols))
                Q = np.zeros((volField._Np, volField._Nn))
                W = np.zeros((volField._Nn, volField._Nn))

                # Loop over neighbours Nn and construct matrix Q, each neighbour cell have its row
                # (n-th column of Q is q(xn-x))
                for i in range(Nn[faceI]):
                    # Neighbour cell centre
                    neiC = mesh.C()[molecules[faceI][i]]
                    Q[0,i] = 1.0
                    Q[1,i] = neiC[0]-gpI[0]
                    Q[2,i] = neiC[1]-gpI[1]
                    Q[3,i] = neiC[2]-gpI[2]
                    #print(neiC)
                    # Loop over elements in truncated Taylor series and add their value
                    # for j in range(Np):
                    #     for k in range(Np-j):
                    #
                    #
                    #         Q[j,i] = pow(neiC[0]-gpI[0], j) * pow(neiC[1]-gpI[1], k) \
                    #             * (1.0/(np.math.factorial(j) * np.math.factorial(k)))
                    #         print(Q[j,i])

                # Loop over neighbours Nn and find max distance
                rs = 0.0
                for i in range(Nn[faceI]):
                    neiC = mesh.C()[molecules[faceI][i]]
                    rsNew = max(rs, np.linalg.norm(gpI - neiC))
                    rs = rsNew

                # Loop over neighbours Nn and construct matrix W (diagonal matrix)
                w_diag = np.zeros(volField._Nn)
                for i in range(Nn[faceI]):
                    neiC = mesh.C()[molecules[faceI][i]]
                    dist = np.linalg.norm(neiC - gpI)
                    w_diag[i] = self.weight(abs(dist), rs)

                if(faceI >= mesh.nInternalFaces()):
                    facePatchType = mesh.facePatchType(faceI, self._volField)

                    if(facePatchType == 'empty'):
                        # Save some time by skipping calculations on empty faces
                        # Fill coefficient vectors with zeros to later on avoid havin problem
                        # when looping through list
                        cx[faceI].append([0.0 for _ in range(volField._Nn)])
                        cy[faceI].append([0.0 for _ in range(volField._Nn)])
                        cz[faceI].append([0.0 for _ in range(volField._Nn)])
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
                cx[faceI].append(list(A[1, :]))
                cy[faceI].append(list(A[2, :]))
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
