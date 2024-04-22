"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume field boundary conditions
"""

import re
import numpy as np

class volFieldBoundaryConditions():

    class updatePatchCoeffs:

        def fixedValue(mesh, patchName, patchValue, boundaryValues):
            print(f"-> updatePatchCoeffs, fixed value: {patchValue}, on {patchName} patch")
            pass

        def empty(self, patchName, patchValue, boundaryValues):
            print(f"-> updatePatchCoeffs, empty patch on {patchName}")


