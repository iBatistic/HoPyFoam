"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['interpolateCellGrad']


import os
import sys

from src.foam.foamFileParser import *
from petsc4py import PETSc

class interpolateCellGrad():

    @classmethod
    def cellToFace(self, psi, gamma, cellPsi, pBar):
        '''
        -----
        '''

        cellGrad = self.cellGrad(self, psi, gamma, cellPsi, pBar)

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
                c = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += \
                            c[j] * cellGrad[cellIndex]

        # Boundary faces
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("interpolateBoundaryConditions." + patchType)
            patchContribution(self, psi, patch, GaussPointsValues, cellGrad, cellPsi, gamma)

        return GaussPointsValues


    def cellGrad(self, psi, mu, cellPsi, pBar):

        mesh = psi._mesh

        cellsGradValue = np.zeros((mesh.nCells, 3))

        # Loop over cells and calculate cell gradient value
        for cellI in range(mesh.nCells):
            cellStencil = pBar._cellsInterpolationMolecule[cellI]

            # Loop over interpolation stencil
            for i, cellIndex in enumerate(cellStencil):

                cx = pBar.LRE().cellGradCoeffs[cellI][i]
                cellsGradValue[cellI] += cx * np.array(psi._cellValues[cellIndex])

        # Add boundary contribution. Cells at boundary have ghost point at
        # the boundary which is not indexed in cell stencil
        for patch in pBar._boundaryConditionsDict:
            # Get patch type
            patchType = pBar._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("cellGradBoundaryConditions." + patchType)
            patchContribution(self, psi, patch, cellsGradValue, cellPsi, mu, pBar)

        return cellsGradValue


class cellGradBoundaryConditions(interpolateCellGrad):

    def empty(self, *args):
        pass

    def fixedValue(self, psi, patch, cellsGradValue, cellPsi, mu, pBar):
        '''
        This is zero gradient boundary condition for pressure.
        In this case ghost point have zero value of gradient and nothing is
        added to the cellGradValue.

        ili suprotno, izracunaj vrijednost i dodaj je u molekulu i ne pizdi lude.
        cx u value at boundary nije nula! moram izracunat vrijednost
        '''

        # 1. Stage: get pressure at boundary
        mesh = psi._mesh

        GaussPointsPressure = np.zeros((mesh.nFaces, psi.Ng, 1))

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
            faceStencil = pBar._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Value at Gauss point which satisfy zero-grad condition
                gpValue = 0

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cx = pBar.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):
                    gpValue += np.array(psi._cellValues[cellIndex]) * (cx[j] @ nf)

                    # Boundary face centre is not included in face stencil list
                    if j == (len(faceStencil) - 1):
                        gpValue /= -(cx[j + 1] @ nf)

                GaussPointsPressure[faceI][i] = gpValue

        # 2. Stage: add ghost cell contribution
        for faceI in range(startFace, startFace + nFaces):
            cellI = mesh._owner[faceI]

            # Last coefficient is placed on the end
            cx = pBar.LRE().cellGradCoeffs[cellI][-1]

            # In case that cell have 2 boundary faces, coefficient for this
            # bc is placed at -2 position
            if len(pBar._cellsInterpolationMolecule[cellI]) + 2 == psi.Nn:
                cx = pBar.LRE().cellGradCoeffs[cellI][-2]

            # Gauss point at face centre is taken for ghost point
            cellsGradValue[cellI] += cx * GaussPointsPressure[faceI][psi._GaussPointsNb // 2]
        #
        # print(cellsGradValue[325])
        # print(cellsGradValue[326])
        # print(cellsGradValue[327], "\n")

    def pressureTraction(self, psi, patch, cellsGradValue, cellPsi, mu, pBar):
        '''
        This is fixed value pressure from constitutive relation
        '''

        # 1. Stage: get pressure at boundary
        mesh = psi._mesh
        GaussPointsPressure = np.zeros((mesh.nFaces, psi.Ng, 1))

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = cellPsi._facesGaussPointsAndWeights

        # Prescribed traction value at boundary
        prescribedValue = cellPsi._boundaryConditionsDict[patch]['value']['uniform']

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = cellPsi._facesInterpolationMolecule[faceI]

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
                    nGrad = (mu * ((cxSecondPsi[j] @ nf) * I)) @ cellPsi._cellValues[cellIndex]

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
                    nGradT = mu * (T @ cellPsi._cellValues[cellIndex])

                    gpValue += nf @ (nGrad + nGradT - prescribedValue)

                GaussPointsPressure[faceI][i] = -gpValue

        # 2. Stage: add ghost cell contribution
        for faceI in range(startFace, startFace + nFaces):
            cellI = mesh._owner[faceI]

            # Last coefficient is placed on the end
            cx = pBar.LRE().cellGradCoeffs[cellI][-1]

            # Gauss point at face centre is taken for ghost point
            cellsGradValue[cellI] += cx * GaussPointsPressure[faceI][pBar._GaussPointsNb // 2]


class interpolateBoundaryConditions(interpolateCellGrad):

    def empty(self, *args):
        pass

   # def fixedValue(self, psi, patch, GaussPointsValues,cellGrad, cellPsi, mu):
    #    return interpolateBoundaryConditions.fixedValueFromZeroGrad(self, psi, patch, GaussPointsValues, cellGrad, cellPsi, mu)

    def fixedValueFromZeroGrad(self, psi, patch, GaussPointsValues, cellGrad, cellPsi, mu):
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

        # Loop over patch faces
        # for faceI in range(startFace, startFace + nFaces):
        #
        #     # List of face Gauss points [1] and weights [0]
        #     faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]
        #
        #     # Current face points interpolation stencil
        #     faceStencil = psi._facesInterpolationMolecule[faceI]
        #
        #     # Loop over Gauss points
        #     for i, gp in enumerate(faceGaussPointsAndWeights[1]):
        #         # Interpolation coeffs
        #        # c = psi.LRE().coeffs()[faceI][i]
        #         c = psi.LRE().coeffs()[faceI][i]
        #
        #         for j, cellIndex in enumerate(faceStencil):
        #             GaussPointsValues[faceI][i] += cellGrad[cellIndex] * c[j]


    def pressureTraction(self, psi, patch, GaussPointsValues, cellGrad, cellPsi, mu):
        #pass
        #warnings.warn(f"pressureTraction at fixedValueFromZeroGrad is ???...\n", stacklevel=3)

        # '''
        # Calculate the pressure from Hooke's law. This is practically a fixed
        # value, with value from the constitutive relation.
        # '''
        mesh = psi._mesh

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        #
        # # Prescribed traction value at boundary
        # prescribedValue = cellPsi._boundaryConditionsDict[patch]['value']['uniform']
        #
        # # 1 stage: Get boundary face pressure from constitutive relation
        # GaussPointsPressure = np.zeros((mesh.nFaces, psi.Ng, 1))
        #
        # # Loop over patch faces
        # for faceI in range(startFace, startFace + nFaces):
        #
        #     # Face normal
        #     nf = mesh.nf[faceI]
        #
        #     # List of face Gauss points [1] and weights [0]
        #     faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]
        #
        #     # Current face points interpolation stencil
        #     faceStencil = psi._facesInterpolationMolecule[faceI]
        #
        #     # Loop over Gauss points
        #     for i, gp in enumerate(faceGaussPointsAndWeights[1]):
        #         # Value at Gauss point which satisfy zero-grad condition
        #         gpValue = 0
        #
        #         # Gauss point interpolation coefficient vector for each neighbouring cell
        #         cxSecondPsi = cellPsi.LRE().gradCoeffs()[faceI][i]
        #
        #         # Loop over Gauss point interpolation stencil and add
        #         # stencil cells contribution to matrix
        #         for j, cellIndex in enumerate(faceStencil):
        #             # Normal face dot with gradient at Gauss point
        #             I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=PETSc.ScalarType)
        #             nGrad =  ( mu * ((cxSecondPsi[j] @ nf) * I)) @ cellPsi._cellValues[cellIndex]
        #
        #             # Normal face dot with gradient transpose at Gauss point
        #             nfx = nf[0]
        #             nfy = nf[1]
        #             nfz = nf[2]
        #             cxx = cxSecondPsi[j][0]
        #             cxy = cxSecondPsi[j][1]
        #             cxz = cxSecondPsi[j][2]
        #             T = np.array([[cxx * nfx, cxx * nfy, cxx * nfz],
        #                                  [cxy * nfx, cxy * nfy, cxy * nfz],
        #                                  [cxz * nfx, cxz * nfy, cxz * nfz]], dtype=PETSc.ScalarType)
        #             nGradT = mu*(T @ cellPsi._cellValues[cellIndex])
        #
        #             gpValue += nf @ (nGrad  + nGradT - prescribedValue)
        #
        #         GaussPointsPressure[faceI][i] = -gpValue
        #
        # # 2 stage: Calculate boundary gradient, previously calculated pressure
        # # is included in calculation as boundary face is part of the molecule
        # GaussPointsGrad = np.zeros((mesh.nFaces, psi.Ng, 3))
        #
        # # Loop over patch faces
        # for faceI in range(startFace, startFace + nFaces):
        #
        #     # List of face Gauss points [1] and weights [0]
        #     faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]
        #
        #     # Current face points interpolation stencil
        #     faceStencil = psi._facesInterpolationMolecule[faceI]
        #
        #     # Loop over Gauss points
        #     for i, gp in enumerate(faceGaussPointsAndWeights[1]):
        #
        #         # Gauss point interpolation coefficient vector for each neighbouring cell
        #         cx = psi.LRE().gradCoeffs()[faceI][i]
        #
        #         # Loop over Gauss point interpolation stencil and add
        #         # stencil cells contribution to matrix
        #         for j, cellIndex in enumerate(faceStencil):
        #             GaussPointsGrad[faceI][i] += cx[j] * np.array(psi._cellValues[cellIndex])
        #
        #             # Boundary face centre is not included in face stencil list
        #             if j == (len(faceStencil) - 1):
        #                 GaussPointsGrad[faceI][i] += cx[j+1] * GaussPointsPressure[faceI][i]

        # print(GaussPointsGrad[startFace:startFace+nFaces])

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
                c = cellPsi.LRE().coeffs()[faceI][i]

                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += cellGrad[cellIndex] * c[j]

                #Boundary face centre is not included in face stencil list
                #    if j == (len(faceStencil) - 1):
                #        GaussPointsValues[faceI][i] += c[j + 1] * GaussPointsGrad[faceI][i]


        warnings.warn(f"\n Gore cell values mi nije dobro!\n", stacklevel=3)