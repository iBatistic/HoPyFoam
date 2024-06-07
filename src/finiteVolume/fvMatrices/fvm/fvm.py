"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description

"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'

from src.finiteVolume.fvMatrices.fvm.operators import *
from src.finiteVolume.fvMatrices.fvMatrix import fvMatrix

class fvm(fvMatrix):

    @classmethod
    def defineMatrix(self, psi, operatorName, gamma):

        # Make function for Laplacian, LaplacianTrace and LaplacianTranspose
        if (operatorName[:9] == "Laplacian"):
            operator = eval("LaplacianOperator." + operatorName)
        else:
            raise ValueError("Unknown operator: " + operatorName)

        # Operator function returns A,b
        return operator(psi, gamma)
