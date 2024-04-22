"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume scalar field class
"""

from src.foam.field.volField import volField

class volScalarField(volField):

    # Dimensions of scalar field
    _dimensions = 1

    # Number of integration points per CV face
    _GaussPointsNb = 1

    # Number of terms in Taylor expansion
    _Np  = 4

    def __init__(self, fieldName, mesh, boundaryAndInitialConditions):
        super().__init__(fieldName, mesh, boundaryAndInitialConditions)
        self._dimensions = 1
        self._GaussPointsNb = 1
        self._Np = 4


    def test(self):
        pass