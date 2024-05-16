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

import numpy as np

class fvMatrix():

    def __init__(self, psi, source, A):
        self._A = A
        self._source = source
        self._psi = psi

    @classmethod
    def construct(self, psi, operator, gamma):

        source, A = self.defineMatrix(psi, operator, gamma)

        return self(psi, source, A)

    def solve(self):
        print('Solving system of equations\n')
        self._psi._cellValues = np.linalg.solve(self._A, self._source)

        # Relative error, L_2 and L_infinity norm
        avgRelError = 0.0
        maxRelError = 0.0
        L2 = 0.0
        Linf = 0.0

        # Short code to check error on example_2
        maxSolVal = 0.0
        for cellI in range(self._psi._mesh.nCells()):
            x = self._psi._mesh.C()[cellI][0]
            y = self._psi._mesh.C()[cellI][1]

            sol = abs(x*x - y*y)
            if sol > maxSolVal:
                maxSolVal = sol

        for cellI in range(self._psi._mesh.nCells()):


            x = self._psi._mesh.C()[cellI][0]
            y = self._psi._mesh.C()[cellI][1]

            #analySol = x*x - y*y + 1e-10
            analySol = np.sin(1*y) * np.exp(1*x)
            diff = abs(self._psi._cellValues[cellI] - analySol)[0]
            #print(diff, analySol, x, y)
            relError = 100 * abs(diff / maxSolVal)
            avgRelError += relError

            if(relError > maxRelError):
               maxRelError = relError

            L2 += np.math.pow(diff, 2)

            if(diff > Linf):
                Linf = diff

        L2 = np.math.sqrt(L2/self._psi._mesh.nCells())
        avgRelError /= self._psi._mesh.nCells()

        print(f'\nAverage relative error: {avgRelError:.4e} %')
        print(f'Maximal relative error: {maxRelError:.4e} %')
        print(f'L_2 error: {L2:.4e}')
        print(f'L_infinty error: {Linf:.4e}')