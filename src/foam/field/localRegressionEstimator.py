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
import warnings
import matplotlib.pyplot as plt

from src.foam.decorators import timed

class localRegressionEstimator():

    # Shape parameter of the kernel
    # Value of this constant is taken from Castrillo et al. paper
    _k = 6

    def __init__(self, volField, mesh):
        self._volField = volField
        self._mesh = mesh

        # Gauss faces points LRE coefficients
        self._c, self._cx, self._cy, self._cz = self.makeFaceCoeffs(volField, mesh, "QR")

        # Cell centre points LRE coefficients
        self._cCell, self._cxCell, self._cyCell, self._czCell = self.makeCellCoeffs(volField, mesh)

        # Gauss faces points final interpolation coefficients, assembled using cx,cy and cz
        # Each item in list is corresponding face, each face has sublist of Gauss points
        # Each Gauss point has interpolation vector (cx, cy, cz)
        self._C = self.makeFaceCoeffsVector()

        self._CCell = self.makeCellCoeffsVector()

    # Return gradCoeffs vector cx for each Gauss points
    def makeFaceCoeffsVector(self):

        coeffs = [[[] for i in range (self._volField._GaussPointsNb)] for x in range(self._mesh.nFaces)]

        for faceI in range(self._mesh.nFaces):
            for point in range(self._volField._GaussPointsNb):
                for cellN in range(self._volField.Nn):
                    coeffs[faceI][point].append(np.array(\
                        [self._cx[faceI][point][cellN],\
                         self._cy[faceI][point][cellN],\
                         self._cz[faceI][point][cellN]]))
        return coeffs

    def makeCellCoeffsVector(self):
        coeffs = [[] for x in range(self._mesh.nCells)]
        #print(self._cxCell[0][1])
        warnings.warn(f"makeCellCoeffsVector treba napraviti kako spada"
                      f"sad bi moglo ovako raditi ali nije provjereno.\n", stacklevel=3)

        for cellI in range(self._mesh.nCells):
            for cellN in range(self._volField.Nn):
                #print(cellI, cellN)
                coeffs[cellI].append(np.array( \
                    [self._cxCell[cellI][0][cellN], \
                     self._cyCell[cellI][0][cellN], \
                     self._czCell[cellI][0][cellN]]))

        #print(coeffs[0][0])
        return coeffs

    def gradCoeffs(self):
        return self._C

    @property
    def cellGradCoeffs(self):
        return self._CCell

    def coeffs(self):
        return self._c

    @timed
    def makeFaceCoeffs(self, volField, mesh, method) -> list[list[np.ndarray]]:

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
        c =  [[] for x in range(mesh.nFaces)]
        cx = [[] for x in range(mesh.nFaces)]
        cy = [[] for x in range(mesh.nFaces)]
        cz = [[] for x in range(mesh.nFaces)]

        # List of matrix M condition numbers
        condNumbers = []

        # Face Gauss points
        facesGaussPoints = [sublist[1] for sublist in volField._facesGaussPointsAndWeights]

        # Loop over Gauss points of interior faces
        for faceI in range(mesh.nFaces):

            # Loop over face neighbours Nn and find max distance
            rs = 0.0
            for i in range(Nn[faceI]):
                neiC = mesh.C[molecules[faceI][i]]
                rsNew = max(rs, np.linalg.norm(mesh.Cf[faceI] - neiC))
                rs = rsNew

            # Loop over face Gauss points
            for gpI in facesGaussPoints[faceI]:

                # np.zeros((rows, cols))
                Q = np.zeros((volField.Np, volField.Nn))
                W = np.zeros((volField.Nn, volField.Nn))

                # Loop over neighbours Nn and construct matrix Q, each neighbour cell have its row
                # (n-th column of Q is q(xn-x))
                for i in range(Nn[faceI]):
                    # Neighbour cell centre
                    neiC = mesh.C[molecules[faceI][i]]

                    # Taylor series terms calculated using nested for loops
                    N = volField.N

                    # Scaling factor
                    h = 2.0 * rs
                    pos = int(0)
                    for I in range(N+1):
                        for J in range(N-I+1):
                            fact = np.math.factorial(I) * np.math.factorial(J)
                            Q[pos, i] = pow(neiC[0]-gpI[0], I) * pow(neiC[1]-gpI[1], J) * (1.0/fact)# * (1/pow(h,I+J))
                            pos += 1;

                # Loop over neighbours Nn and construct matrix W (diagonal matrix)
                w_diag = np.zeros(volField.Nn)
                for i in range(Nn[faceI]):
                    neiC = mesh.C[molecules[faceI][i]]
                    dist = np.linalg.norm(neiC - gpI)
                    w_diag[i] = self.weight(abs(dist), rs)

                if(faceI >= mesh.nInternalFaces):
                    facePatchType = mesh.facePatchType(faceI, self._volField)

                    if (facePatchType == 'empty'):
                        # Save some time by skipping calculations on empty faces
                        # Fill coefficient vectors with zeros to later on avoid having problem
                        # when looping through list
                        c[faceI].append([0.0 for _ in range(volField.Nn)])
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

                if (method == "INV"):
                    # Calculate matrix M = Q*W*Q^T
                    M = Q @ W @ Q.T

                    if (M.shape[0] != M.shape[1] | M.shape[0] == Np):
                        raise ValueError("Matrix M should be Np x Np shape. Something went wrong!")

                    # Calculate matrix A = M^-1*Q*W
                    # Depending on condition number choose pinv or inv
                    condNumber = np.linalg.cond(M)
                    if condNumber < 1e5:
                        # Matrix is not close to singular, use np.linalg.inv()
                        A = np.linalg.inv(M) @ Q @ W
                    else:
                        # Matrix is close to singular, use np.linalg.pinv()
                        A = np.linalg.pinv(M) @ Q @ W

                elif (method == "QR"):

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
                else:
                    print("Invalid method for matrix A calculation")
                    sys.exit(1)


                if condNumber > 1e12:
                    warnings.warn(f"At least one face with large condition number (>1e12) detected\n "
                                  f"This will mess up results. Terminating calculation...\n", stacklevel=3)


                # Interpolation coefficients for the current Gauss point
                # With above for loop Taylor expression looks like:
                # 1, (y-b), (y-b)^2, (x-a), (x-a)(x-b), (x-a)^2
                c[faceI].append(list(A[0, :]))
                cx[faceI].append(list(A[1+N, :]))
                cy[faceI].append(list(A[1, :]))
                cz[faceI].append([0.0 for _ in range(len(list(A[1, :])))])

                # Store Gauss point condition number
                condNumbers.append(condNumber)

                # Used for debugging
                # Write matrices to textual file, so they can be checked
                #self.writeData(faceI, gpI, Q, W, condNumber, A, fileName)

        # Plot distribution of condition number
        plotCondNumbers(condNumbers)

        return c, cx, cy, cz

    def makeCellCoeffs(self, volField, mesh) -> list[list[np.ndarray]]:
        print("sto sa ghost celijama na rubu koje nisu proracunska tocka jaoo komplik")

        # Cell interpolation molecules
        molecules = volField._cellsInterpolationMolecule

        # Stencils size (this is equal to volField.Nn only at interior cells)
        Nn = [len(molecule) for molecule in molecules]

        # Number of terms in Taylor expression
        Np = volField.Np

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

                # Taylor series terms calculated using nested for loops
                N = volField.N

                # Scaling factor
                h = 2.0 * rs
                pos = int(0)
                for I in range(N + 1):
                    for J in range(N - I + 1):
                        fact = np.math.factorial(I) * np.math.factorial(J)
                        Q[pos, i] = pow(neiC[0] - mesh.C[cellI][0], I) * pow(neiC[1] - mesh.C[cellI][1], J) * (
                                    1.0 / fact)  # * (1/pow(h,I+J))
                        pos += 1;

            # Loop over neighbours Nn and construct matrix W (diagonal matrix)
            w_diag = np.zeros(volField.Nn)
            for i in range(Nn[cellI]):
                neiC = mesh.C[molecules[cellI][i]]
                dist = np.linalg.norm(neiC - mesh.C[cellI])
                w_diag[i] = self.weight(abs(dist), rs)

            # Add boundary face contribution to Q and W
            warnings.warn(f"Add boundary face contribution to Q and W ???...\n", stacklevel=3)
            if cellI in mesh.boundaryCells:
                cellFaces = mesh.cellFaces[cellI]
                for faceI in cellFaces:
                    facePatchType = mesh.facePatchType(faceI, self._volField, False)

                    if Nn[cellI] == volField.Nn - 1:
                        if facePatchType != 'empty' and facePatchType is not None:
                            w_diag[-1] = 1.0
                            Q[0][-1] = 1.0
                    if Nn[cellI] == volField.Nn - 2:
                        if facePatchType == "pressureTraction":
                            w_diag[-1] = 1.0
                            Q[0][-1] = 1.0
                        if facePatchType == "fixedValueFromZeroGrad":
                            w_diag[-2] = 1.0
                            Q[0][-2] = 1.0

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

            if condNumber > 1e12:
                warnings.warn(f"At least one face with large condition number (>1e12) detected\n "
                              f"This will mess up results. Terminating calculation...\n", stacklevel=3)

            # Interpolation coefficients for the current Gauss point
            # With above for loop Taylor expression looks like:
            # 1, (y-b), (y-b)^2, (x-a), (x-a)(x-b), (x-a)^2
            c[cellI].append(list(A[0, :]))
            cx[cellI].append(list(A[1 + N, :]))
            cy[cellI].append(list(A[1, :]))
            cz[cellI].append([0.0 for _ in range(len(list(A[1, :])))])

        return c, cx, cy, cz

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
    def writeData(self, faceI, gpI, Q, W, condNumber, A, fileName):

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
        #file.write(f'Interpolation molecule: {mol} \n')
        #file.write(f'Neigbours centres: {neis} \n')
        file.write(f"Matrix Q:\n {Q}\n")
        file.write(f"Matrix W:\n {W}\n")
        #file.write(f"Matrix M:\n {M}\n")
        file.write(f"Matrix M condition number:\n {condNumber}\n")
        file.write(f"Matrix A:\n {A}\n \n")
        file.close()

def plotCondNumbers(condNumbers):
    # Plot a histogram to visualize the distribution of condition numbers
    plt.figure(figsize=(10, 6))
    plt.hist(condNumbers, bins=50, edgecolor='black', alpha=0.7)
    plt.suptitle('Distribution of Matrix M Condition Numbers')
    plt.title('Data is divided into 50 bins', fontsize=8)
    plt.xlabel('Condition Number')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True)
    figure = plt.gcf()
    figure.savefig("conditionNumbers.png", dpi=100)
    plt.close(figure)
