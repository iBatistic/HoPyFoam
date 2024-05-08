"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume scalar field class
"""

import os
import shutil

from src.foam.field.volField import volField
import numpy as np
from src.foam.argList import BANNER

class volScalarField(volField):

    # Dimensions of scalar field
    _dimensions = 1

    # Number of integration points per CV face
    _GaussPointsNb = 1

    # Number of terms in Taylor expansion
    _Np  = 4

    # Number of cells in interpolation stencil
    _Nn = 4

    def __init__(self, fieldName, mesh, boundaryAndInitialConditions):
        super().__init__(fieldName, mesh, boundaryAndInitialConditions)

    def write(self, timeValue):
        print(f"Writing {self._fieldName} field fot time {timeValue} s\n")

        filePath = str(timeValue) + "/" + self._fieldName

        # Remove old one if exists
        if (os.path.exists(str(timeValue))):
            shutil.rmtree(str(timeValue))

        os.mkdir(str(timeValue))

        # Write field in OF format
        with open(filePath, 'w') as file:
            # Write data to the file
            file.write('/*\n' + BANNER + '*/\n')
            file.write(self.foamFile('volScalarField', self._fieldName, timeValue))
            file.write("\ndimensions      [0 0 0 1 0 0 0]; \n\n")
            file.write("internalField   nonuniform List<scalar>\n")
            file.write(f"{self._mesh.nCells()}\n")
            file.write("(\n")
            for i in range(self._mesh.nCells()):
                file.write(f'{self._cellValues[i][0]} \n')
            file.write(")\n")
            file.write(";\n\n")


            file.write('boundaryField\n{\n')
            for patch in self._boundaryConditionsDict:
                patchName = str(patch)
                patchType = self._boundaryConditionsDict[patchName]['type']
                file.write(f'\t{patchName}\n\t{{\n\t\ttype\t{patchType};\n\t}}\n')
            file.write('}\n')