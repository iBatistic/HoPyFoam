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
__all__ = ['volFieldBoundaryConditions']

import re
import numpy as np
import sys

class volFieldBoundaryConditions():

    class LREcoeffs:

        def fixedValue(w_diag, Q):
            w_diag[-1] = 1.0
            Q[0][-1] = 1.0

        def empty(w_diag, Q):
            pass

        def zeroGradient(w_diag, Q):
            pass

        def solidTraction(w_diag, Q):
            pass

        def fixedValueFromZeroGrad(w_diag, Q):
            volFieldBoundaryConditions.LREcoeffs.zeroGradient(w_diag, Q)

        def pressureTraction(w_diag, Q):
            volFieldBoundaryConditions.LREcoeffs.fixedValue(w_diag, Q)

    # Evaluate is used to calculate values at boundaries after
    # cell centred values are calculated
    class evaluate:

        def empty(*args):
            pass

        def zeroGradient(*args):
            pass

        def pressureTraction(*args):
            pass

        def fixedValue(psi, mesh, bdDict, patchName, boundaryValues):
            boundary = mesh._boundary
            nInternalFaces = mesh.nInternalFaces

            patchValue = bdDict[patchName]['value']['uniform']

            for patch in boundary:
                if patch == patchName:
                    startFace = boundary[patch]['startFace']
                    nFaces = boundary[patch]['nFaces']

                    for faceI in range(nFaces):
                        bIndex = startFace + faceI - nInternalFaces
                        boundaryValues[bIndex] = patchValue

        def solidTraction(psi, mesh, bdDict, patchName, boundaryValues):
            boundary = mesh._boundary
            nInternalFaces = mesh.nInternalFaces

            for patch in boundary:
                if patch == patchName:
                    startFace = boundary[patch]['startFace']
                    nFaces = boundary[patch]['nFaces']

                    for faceI in range(startFace, startFace + nFaces):

                        # Reset boundary value to avoid accumulation
                        boundaryValues[faceI- nInternalFaces] = np.array([0,0,0])

                         # Current face points interpolation stencil
                        faceStencil = psi._facesInterpolationMolecule[faceI]

                        # One Gauss point coincide with face centre if Ng is odd
                        # We use interpolation data of that point to calculate
                        # displacement at boundary face centre
                        if psi._GaussPointsNb % 2 != 0:
                            # Gauss point interpolation coefficient vector
                            c = psi.LRE().boundaryCoeffs[faceI - nInternalFaces][psi._GaussPointsNb // 2]

                            for j, cellIndex in enumerate(faceStencil):
                                boundaryValues[faceI- nInternalFaces] += np.array(psi._cellValues[cellIndex]) * c[j]

        def fixedValueFromZeroGrad(psi, mesh, bdDict, patchName, boundaryValues):
            pass
            # boundary = mesh._boundary
            # GaussPointsAndWeights = psi._facesGaussPointsAndWeights
            #
            # for patch in boundary:
            #     if patch == patchName:
            #         startFace = boundary[patch]['startFace']
            #         nFaces = boundary[patch]['nFaces']
            #
            #         for faceI in range(startFace, startFace + nFaces):
            #
            #             # Reset boundary value to avoid accumulation
            #             boundaryValues[faceI] = 0
            #
            #             nf = mesh.nf[faceI]
            #
            #             # List of face Gauss points [1] and weights [0]
            #             faceGaussPoints = GaussPointsAndWeights[faceI][1]
            #
            #             # Current face points interpolation stencil
            #             faceStencil = psi._facesInterpolationMolecule[faceI]
            #
            #             # One Gauss point coincide with face centre if Ng is odd
            #             # We use interpolation data of that point to calculate
            #             # displacement at boundary face centre
            #             if psi._GaussPointsNb % 2 != 0:
            #
            #                 # Gauss point interpolation coefficient vector
            #                 cx = psi.LRE().gradCoeffs()[faceI][psi._GaussPointsNb // 2]
            #
            #                 for j, cellIndex in enumerate(faceStencil):
            #                     boundaryValues[faceI] += psi._cellValues[cellIndex] * (cx[j] @ nf)
            #                     #print(boundaryValues[faceI])
            #                     #print(psi._cellValues[cellIndex])
            #                     # Boundary face centre is not included in face stencil list
            #                     if j == (len(faceStencil) - 1):
            #                         boundaryValues[faceI] /= -(cx[j + 1] @ nf)
            #                         print(cx[j + 1] @ nf)
                            #print(boundaryValues[faceI])print(boundaryValues[faceI])
