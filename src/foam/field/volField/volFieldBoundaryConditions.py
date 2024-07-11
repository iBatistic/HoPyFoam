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

    # Evaluate is used to calculate values at boundaries after
    # cell centred values are calculated
    class evaluate:

        def empty(*args):
            pass

        def zeroGradient(*args):
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
            GaussPointsAndWeights = psi._facesGaussPointsAndWeights

            for patch in boundary:
                if patch == patchName:
                    startFace = boundary[patch]['startFace']
                    nFaces = boundary[patch]['nFaces']

                    for i in range(nFaces):
                        faceI = startFace + i - nInternalFaces

                        # List of face Gauss points [1] and weights [0]
                        faceGaussPoints = GaussPointsAndWeights[faceI][1]

                        # Current face points interpolation stencil
                        faceStencil = psi._facesInterpolationMolecule[startFace + i]

                        # One Gauss point coincide with face centre if Ng is odd
                        # We use interpolation data of that point to calculate
                        # displacement at boundary face centre
                        if psi._GaussPointsNb % 2 != 0:
                            # Gauss point interpolation coefficient vector
                            c = psi.LRE().coeffs()[startFace + i][psi._GaussPointsNb // 2]

                            for j, cellIndex in enumerate(faceStencil):
                                boundaryValues[faceI] += np.array(psi._cellValues[cellIndex]) * c[j]

