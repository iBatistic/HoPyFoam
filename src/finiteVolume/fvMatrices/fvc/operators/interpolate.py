"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['interpolate']


import os
from src.foam.foamFileParser import *
from petsc4py import PETSc

class interpolate():

    @classmethod
    def diagToFace(self, diagTensor, U):
        '''
        Interpolate momentum matrix diagonal from cell centres to face
        Gauss points. All boundaries are extrapolated.
        '''
        mesh = U._mesh

        # Initialize faces Gauss points values
        GaussPointsValues = np.zeros((mesh.nFaces, U.Ng, 3, 3))

        for faceI in range(mesh.nFaces):
            faceStencil = U._facesInterpolationMolecule[faceI]
            faceGaussPointsAndWeights = U._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Gauss point interpolation coefficient vector
                c = U.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += \
                        diagTensor[cellIndex] * c[j]

        return GaussPointsValues

    @classmethod
    def cellToFace(self, psi, gamma=None, cellPsi=None):
        '''
        Interpolate value from cell centres to face Gauss points. Special care
        is given at boundaries.
        '''

        mesh = psi._mesh

        # Initialize faces Gauss points values
        GaussPointsValues = np.zeros((mesh.nFaces, psi.Ng, psi.dim))

        # Interior faces
        for faceI in range(mesh.nInternalFaces):
            faceStencil = psi._facesInterpolationMolecule[faceI]
            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Gauss point interpolation coefficient vector
                c = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += \
                            np.array(psi._cellValues[cellIndex]) * c[j]

        # Enforce boundary conditions at boundary faces
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("interpolateBoundaryConditions." + patchType)
            patchContribution(self, psi, patch, GaussPointsValues, cellPsi, gamma)

        return GaussPointsValues


class interpolateBoundaryConditions(interpolate):

    def empty(self, *args):
        pass

    # Hmmm da li je ovo dobro ili tamo kada radim loop za gradijent moram zanemarit ovaj face...
    def zeroGradient(self, psi, patch, GaussPointsValues, cellPsi, gamma):
        pass
        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Interpolation coeffs
                # c = psi.LRE().coeffs()[faceI][i]
                #
                # for j, cellIndex in enumerate(faceStencil):
                #     GaussPointsValues[faceI][i] += np.array(psi._cellValues[cellIndex]) * c[j]

                GaussPointsValues[faceI][i] = psi._cellValues[mesh._owner[faceI]]

    def fixedValue(self, psi, patch, GaussPointsValues, *args):
        '''
        Value is prescribed at this boundary.
        '''
        prescribedValue = psi._boundaryConditionsDict[patch]['value']['uniform']

        startFace = psi._mesh.boundary[patch]['startFace']
        nFaces = psi._mesh.boundary[patch]['nFaces']

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):
            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                GaussPointsValues[faceI][i] = prescribedValue

    def solidTraction(self, psi, patch, GaussPointsValues, *args):
        '''
        Value is calculated by interpolation using interpolation molecule
        '''

        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Interpolation coeffs
                c = psi.LRE().coeffs()[faceI][i]

                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += np.array(psi._cellValues[cellIndex]) * c[j]

    def fixedValueFromZeroGrad(self, psi, patch, GaussPointsValues, *args):
        '''
        Calculates the value at boundary which satisfy condition of
        zero-gradient.
        '''
        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        warnings.warn(f"fixedValueFromZeroGrad is first order accurate currently...\n", stacklevel=3)
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
                cx = psi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                # for j, cellIndex in enumerate(faceStencil):
                #     gpValue += np.array(psi._cellValues[cellIndex]) * (cx[j] @ nf)
                #
                #     # Boundary face centre is not included in face stencil list
                #     if j == (len(faceStencil) - 1):
                #         gpValue /= -(cx[j + 1] @ nf)

                GaussPointsValues[faceI][i] = psi._cellValues[mesh._owner[faceI]]#gpValue

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
                cx = psi.LRE().gradCoeffs()[faceI][i]
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

                GaussPointsValues[faceI][i] = -gpValue

            #print(GaussPointsValues[faceI][3])