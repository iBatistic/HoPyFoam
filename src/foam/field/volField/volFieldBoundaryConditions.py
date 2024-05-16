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

        def analyticalFixedValue(w_diag, Q):
            volFieldBoundaryConditions.LREcoeffs.fixedValue(w_diag, Q)

        def empty(patchName, patchValue, boundaryValues):
            pass


    class evaluate:

        def fixedValue(mesh, patchName, patchValue, boundaryValues, dimensions):
            boundary = mesh._boundary
            nInternalFaces = mesh.nInternalFaces()

            for patch in boundary:
                if patch == patchName:
                    startFace = boundary[patch]['startFace']
                    nFaces = boundary[patch]['nFaces']

                    for faceI in range(nFaces):
                        bIndex = startFace + faceI - nInternalFaces
                        for cmpt in range(dimensions):
                            boundaryValues[cmpt][bIndex] = patchValue#[cmpt]

        def analyticalFixedValue(mesh, patchName, patchValue, boundaryValues, dimensions):
            volFieldBoundaryConditions.evaluate.fixedValue(mesh, patchName, patchValue, boundaryValues, dimensions)

        def empty(mesh, patchName, patchValue, boundaryValues, dimensions):
            pass
