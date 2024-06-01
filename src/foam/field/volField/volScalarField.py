"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume scalar field class
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['volScalarField']

import os
import shutil
import warnings

from src.foam.field.volField import volField
from src.foam.argList import BANNER
from src.foam.foamFileParser import convert_to_float

class volScalarField(volField):

    # Field dimensions
    _dimensions = 1

    def __init__(self, fieldName, mesh, boundaryAndInitialConditions):

        # Number of integration points per CV face
        self._GaussPointsNb = 7

        # Order of interpolation (1 = linear, 2 = quadratic, 3 = qubic...)
        self._N = 3

        # Number of terms in Taylor expansion
        # Example: Second order (quadratic interpolation) has 6 terms in 2D case
        # Example: First order (linear interpolation) has 3 terms in 2D case
        self. _Np  = self.TaylorTermsNumber(self._N, mesh.twoD)

        # Number of cells in interpolation stencil
        self._Nn = 16

        # A quick check for stencil size requirement
        if (self._Np >= self._Nn):
            warnings.warn(f"\nNumber of neighbours {self._Nn} is smaller than number "
                          f"of Taylor expansion terms {self._Np}\n", stacklevel=3)

        # Initialise volField
        super().__init__(fieldName, mesh, boundaryAndInitialConditions)


    # Returns number of terms in Taylor expression, hard-coded
    @classmethod
    def TaylorTermsNumber(self, N, twoD) -> int:

        if (twoD):
            if (N == 1):
                return 3
            elif (N == 2):
                return 6
            elif (N == 3):
                return 10
            else:
                ValueError("Not implemented")
        else:
            ValueError("Not implemented")

    # Return number of terms in Taylor expression
    # For two-dimensional cases we are ignoring terms related to z coordinate!
    @property
    def Np(self) -> int:
        return self._Np

    # Return number of neighbours for this field type
    @property
    def Nn(self) -> int:
        return self._Nn

    # Interpolation  order, polynomial degree
    @property
    def N(self) -> int:
        return self._N

    # Write field in OpenFOAM format
    def write(self, timeValue):
        print(f"\nWriting {self._fieldName} field fot time {timeValue} s\n")

        filePath = str(timeValue) + "/" + self._fieldName

        # Remove old time directory if exists
        if (os.path.exists(str(timeValue))):
            shutil.rmtree(str(timeValue))

        # Make new time directory
        os.mkdir(str(timeValue))

        # Write field in OF format
        with open(filePath, 'w') as file:
            # Write file banner
            file.write('/*\n' + BANNER + '*/\n')
            file.write(self.foamFileDict('volScalarField', self._fieldName, timeValue))
            file.write("\ndimensions      [0 0 0 1 0 0 0]; \n\n")

            # Write internal field
            file.write("internalField   nonuniform List<scalar>\n")
            file.write(f"{self._mesh.nCells}\n")
            file.write("(\n")
            for i in range(self._mesh.nCells):
                file.write(f'{self._cellValues[i]} \n')
            file.write(")\n")
            file.write(";\n\n")

            # Write boundary field
            file.write('boundaryField\n{\n')
            for patch in self._boundaryConditionsDict:
                patchType = self._boundaryConditionsDict[patch]['type']
                file.write(f'\t{patch}\n\t{{\n\t\ttype\t{patchType};\n')
                #
                if patchType == 'fixedValue' or patchType == 'analyticalFixedValue':
                    value = convert_to_float(self._boundaryConditionsDict[patch]['value']['uniform'])
                    file.write(f'\t\tvalue\tuniform {value};\n')

                    if patchType == 'analyticalFixedValue':
                        warnings.warn(f"At boundaries analyticalFixedValue has some dummy value from 0\n "
                                      f"This can mess up postprocessing, however to write exact values "
                                      f"at face centre integration\n should be performed at each boundary "
                                      f"face using values at Gauss points, This can be added later if needed...\n", stacklevel=3)

                file.write(f'\t}}\n')

            file.write('}\n')