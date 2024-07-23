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

    def __init__(self, fieldName, mesh, scalarFieldEntries, N=3):

        # Initialise volField
        super().__init__(fieldName, mesh, scalarFieldEntries, N)

    @property
    def dim(self) -> int:
        return self._dimensions

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
            file.write(f"{self._mesh.nCells}\n(\n")

            for i in range(self._mesh.nCells):
                file.write(f'{self._cellValues[i][0]} \n')
            file.write(")\n;\n\n")

            # Write boundary field
            file.write('boundaryField\n{\n')
            for patch in self._boundaryConditionsDict:
                patchType = self._boundaryConditionsDict[patch]['type']
                file.write(f'\t{patch}\n\t{{\n\t\ttype\t{patchType};\n')

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