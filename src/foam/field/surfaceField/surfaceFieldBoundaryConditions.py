"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume field boundary conditions
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['surfaceFieldBoundaryConditions']

import re
import numpy as np
import warnings
from petsc4py import PETSc

class surfaceFieldBoundaryConditions():

    class evaluateBarGrad:
        """Gradijent imam camo u centrima, tako da na rubovima moram radit
        elstrapolaciju"""
        def empty(self, *args):
            pass

        def pressureTraction(self, psi, psif, cellGrad, patch, GaussPointsValues):
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
                    # Value at Gauss point which satisfy zero-grad condition
                    gpValue = np.array([0.0, 0.0, 0.0])

                    # Gauss point interpolation coefficient vector for each neighbouring cell
                    cx = psi.LRE().boundaryCoeffs[faceI - mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        gpValue += cellGrad[cellIndex] * cx[j]

                    GaussPointsValues[faceI][i] = gpValue


        def zeroGradient(self, psi, psif, cellGrad, patch, GaussPointsValues):
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
                    # Value at Gauss point which satisfy zero-grad condition
                    gpValue = np.array([0.0, 0.0, 0.0])

                    # Gauss point interpolation coefficient vector for each neighbouring cell
                    cx = psi.LRE().boundaryCoeffs[faceI - mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        gpValue += cellGrad[cellIndex] * cx[j]

                    # We need to remove normal component as it is normal zero grad boundary
                    normalGrad = np.dot(gpValue, nf) * nf
                    gpValue -= normalGrad
                    GaussPointsValues[faceI][i] = gpValue

                #     #GaussPointsValues[faceI][i] = cellGrad[mesh._owner[faceI]]
                # if patch == 'right':
                #      print(mesh._owner[faceI])
                #      print("Cell gradient: ", GaussPointsValues[faceI][3])

    class evaluateBarGradCellGrad:

        def empty(self, *args):
            pass

        def zeroGradient(self, psi, psif, patch, cellsGradValue):
            mesh = psi._mesh
            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            for faceI in range(startFace, startFace + nFaces):
                cellI = mesh._owner[faceI]

                # Last coefficient is placed on the end
                cx = psi.LRE().cellGradCoeffs[cellI][-1]

                # In case that cell have 2 boundary faces, coefficient for this
                # bc is placed at -2 position
                if len(psi._cellsInterpolationMolecule[cellI]) + 2 == psi.Nn:
                    cx = psi.LRE().cellGradCoeffs[cellI][-2]

                # Gauss point at face centre is taken for ghost point
                cellsGradValue[cellI] += cx * psif[faceI][psi._GaussPointsNb // 2]

        def pressureTraction(self, psi, psif, patch, cellsGradValue):
            mesh = psi._mesh
            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            for faceI in range(startFace, startFace + nFaces):
                cellI = mesh._owner[faceI]

                # Last coefficient is placed on the end
                cx = psi.LRE().cellGradCoeffs[cellI][-1]

                # Gauss point at face centre is taken for ghost point
                cellsGradValue[cellI] += cx * psif[faceI][psi._GaussPointsNb // 2]

    class evaluateGrad:

        def empty(self, *args):
            pass

        def pressureTraction(self, psi, patch, GaussPointsValues, psif):
            """Tu je pressure fiksiran, znaci moram ga ukljucit u molekulu"""

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
                    # Value at Gauss point which satisfy zero-grad condition
                    gpValue = np.array([0.0, 0.0, 0.0])

                    #Gauss point interpolation coefficient vector for each neighbouring cell
                    cx = psi.LRE().boundaryGradCoeffsGhost[faceI - mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        gpValue += psi._cellValues[cellIndex] * cx[j]

                        # Boundary face centre is not included in face stencil list
                        if j == (len(faceStencil) - 1):
                            gpValue += cx[j + 1] * psif[faceI][i]

                    GaussPointsValues[faceI][i] = gpValue


        def zeroGradient(self, psi, patch, GaussPointsValues, psif):
            """Tu pressure nije fiksiran ali je normalni gradijent. Gradijent
            cu izracunat i samo za normalnu komponentu stavit nulu, nemam drugu ideju"""
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
                    # Value at Gauss point which satisfy zero-grad condition
                    gpValue = np.array([0.0, 0.0, 0.0])

                    # Gauss point interpolation coefficient vector for each neighbouring cell
                    #cx = psi.LRE().boundaryGradCoeffs[faceI - mesh.nInternalFaces][i]
                    # If I use ghost point in which walue is enforced to obey zeroGrad then i
                    # do not need to remove normalGrad
                    cx = psi.LRE().boundaryGradCoeffsGhost[faceI - mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        gpValue += np.array(psi._cellValues[cellIndex]) * cx[j]

                        # Boundary face centre is not included in face stencil list
                        if j == (len(faceStencil) - 1):
                            gpValue += cx[j + 1] * psif[faceI][i]

                    # We need to remove normal component as it is normal zero grad boundary
                    normalGrad = np.dot(gpValue, nf) * nf
                    #gpValue -= normalGrad
                    GaussPointsValues[faceI][i] = gpValue

                # if patch == 'right':
                #     print(mesh._owner[faceI])
                #     #print("Cell gradient: ", psi._cellValues[mesh._owner[faceI]])
                #     print("Boundary face gradient: \n", GaussPointsValues[faceI])

    # Evaluate is used to calculate values at boundaries
    class evaluate:

        def empty(self, *args):
            pass

        #----------------------DISPLACEMENT-----------------------------------#
        # -- This is also used by pressure correction
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
                    c = psi.LRE().boundaryCoeffs[faceI-mesh.nInternalFaces][i]

                    for j, cellIndex in enumerate(faceStencil):
                        GaussPointsValues[faceI][i] += np.array(psi._cellValues[cellIndex]) * c[j]

        #--------------------- PRESSURE----------------------------------#

        def pressureTraction(self, psi, patch, GaussPointsValues, sPsi, mu):
            """
            Calculate the pressure from Hooke's law. This is practically
            a fixed value, with value from the constitutive relation.
            """

            mesh = psi._mesh

            # Preliminaries
            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            GaussPointsAndWeights = psi._facesGaussPointsAndWeights

            # Prescribed traction value at boundary
            prescribedValue = sPsi._boundaryConditionsDict[patch]['value']['uniform']

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
                    cxSecondPsi = sPsi.LRE().boundaryGradCoeffs[faceI-mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        # Normal face dot with gradient at Gauss point
                        I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=PETSc.ScalarType)
                        nGrad = (mu * ((cxSecondPsi[j] @ nf) * I)) @ sPsi._cellValues[cellIndex]

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

                        nGradT = mu * (T @ sPsi._cellValues[cellIndex])

                        gpValue += nf @ (nGrad + nGradT - prescribedValue)

                    GaussPointsValues[faceI][i] = -gpValue

        def zeroGradient(self, psi, patch, GaussPointsValues, *args):
            '''
            Calculates the value at boundary which satisfy condition of
            zero-gradient.
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
                    # Value at Gauss point which satisfy zero-grad condition
                    gpValue = 0

                    # Gauss point interpolation coefficient vector for each neighbouring cell
                    cx = psi.LRE().boundaryGradCoeffsGhost[faceI-mesh.nInternalFaces][i]

                    # Loop over Gauss point interpolation stencil and add
                    # stencil cells contribution to matrix
                    for j, cellIndex in enumerate(faceStencil):
                        gpValue += np.array(psi._cellValues[cellIndex]) * np.dot(cx[j], nf)

                        # Boundary face centre is not included in face stencil list
                        if j == (len(faceStencil) - 1):
                            gpValue /= -np.dot(cx[j + 1], nf)

                    GaussPointsValues[faceI][i] = gpValue
            # print(psi.fieldName)
            # if patch == "right":
            #     for faceI in range(startFace, startFace + nFaces):
            #         for i in range(psi.Ng):
            #             print(GaussPointsValues[faceI][i])
            #
            #         print("cell value:", psi._cellValues[mesh._owner[faceI]])
            #     print("\n")

            #face = startFace + 3
            #print("Usporedba za zeroGradient")
            #print("->",psi._cellValues[mesh._owner[face]])
            #print(GaussPointsValues[face])
                    #print(GaussPointsValues[faceI][i])
                    #print("\n")
                    #GaussPointsValues[faceI][i] = psi._cellValues[mesh._owner[faceI]]gpValue