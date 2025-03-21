"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description

"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.unizg.hr, philip.cardiff@ucd.ie'

from src.finiteVolume.fvMatrices.fvMatrix import fvMatrix
from src.finiteVolume.fvMatrices.fvm.operators import *

class fvm(fvMatrix):

    @classmethod
    def defineMatrix(self, psi, operatorName, gamma, *args):

        # Make function for Laplacian, LaplacianTrace or LaplacianTranspose
        operator = eval(operatorName + "." + "construct")

        # Operator function returns A,b
        return operator(psi, gamma, *args)
