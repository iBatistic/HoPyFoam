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

class localRegressionEstimator:

    _coeffs = "None"

    def __init__(self, volField, mesh):
        self._volField = volField
        self._mesh = mesh
        self._coeffs = self.makeCoeffs(volField, mesh)

    @classmethod
    def makeCoeffs(self, volField, mesh):
        return None

    def updateBoundary(self):
        print('sds')
        #self._boundary = "empty"

    def updateField(self):
        print('sdsdasdsa')
        #self._field = "empty"

