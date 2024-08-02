"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['interpolateGrad']


import os
from src.foam.foamFileParser import *
from petsc4py import PETSc

class interpolateGrad():

    @classmethod
    def cellToFace(self, psi, gamma, cellPsi):
        '''
        -----
        '''
        if psi.dim != 1:
            raise ValueError('cellToFaceGrad implemented only for scalar field')
            sys.exit(1)

        mesh = psi._mesh

        # Initialize faces Gauss points values
        GaussPointsValues = np.zeros((mesh.nFaces, psi.Ng, 3))

        # Interior faces
        for faceI in range(mesh.nInternalFaces):
            faceStencil = psi._facesInterpolationMolecule[faceI]
            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Gauss point grad interpolation coefficient vector
                c = psi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += \
                            c[j] * np.array(psi._cellValues[cellIndex])

        # Enforce boundary conditions at boundary faces
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("interpolateBoundaryConditions." + patchType)
            patchContribution(self, psi, patch, GaussPointsValues, cellPsi, gamma)

        return GaussPointsValues

class interpolateBoundaryConditions(interpolateGrad):

    def empty(self, *args):
        pass

    def fixedValueFromZeroGrad(self, psi, patch, GaussPointsValues, *args):
        '''
        Zero-gradient boundary
        '''
        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Set value to zero-grad
                GaussPointsValues[faceI][i] = np.array([0, 0, 0])

    def pressureTraction(self, psi, patch, GaussPointsValues, cellPsi, mu):
        '''
        Calculate the pressure from Hooke's law. This is practically a fixed
        value, with value from the constitutive relation.
        '''
        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Prescribed traction value at boundary
        prescribedValue = cellPsi._boundaryConditionsDict[patch]['value']['uniform']

        # 1 stage: Get boundary face pressure from constitutive relation
        GaussPointsPressure = np.zeros((mesh.nFaces, psi.Ng, 3))

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Value at Gauss point which satisfy zero-grad condition
                gpValue = 0

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cxSecondPsi = cellPsi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):
                    # Normal face dot with gradient at Gauss point
                    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=PETSc.ScalarType)
                    nGrad =  ( mu * ((cxSecondPsi[j] @ nf) * I)) @ cellPsi._cellValues[cellIndex]

                    # Normal face dot with gradient transpose at Gauss point
                    nfx = nf[0]
                    nfy = nf[1]
                    nfz = nf[2]
                    cxx = cxSecondPsi[j][0]
                    cxy = cxSecondPsi[j][1]
                    cxz = cxSecondPsi[j][2]
                    T = np.array([[cxx * nfx, cxx * nfy, cxx * nfz],
                                         [cxy * nfx, cxy * nfy, cxy * nfz],
                                         [cxz * nfx, cxz * nfy, cxz * nfz]], dtype=PETSc.ScalarType)
                    nGradT = mu*(T @ cellPsi._cellValues[cellIndex])

                    gpValue += nf @ (nGrad  + nGradT - prescribedValue)

            GaussPointsPressure[faceI][i] = gpValue

        # 2 stage: Calculate boundary gradient, previously calculated pressure
        # is included in calculation as boundary face is part of the molecule

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                gpValue = np.array([0, 0, 0])

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cx = psi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i]  += cx[j] * np.array(psi._cellValues[cellIndex])

                    # Boundary face centre is not included in face stencil list
                    if j == (len(faceStencil) - 1):
                        GaussPointsValues[faceI][i] += cx[j+1] * GaussPointsPressure[faceI][i]