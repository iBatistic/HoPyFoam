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
__all__ = ['volVectorField']

import os
import shutil
import warnings

from src.foam.field.volField import volField
from src.foam.argList import BANNER
from src.foam.foamFileParser import convert_to_float

class volVectorField(volField):

    # Field dimensions
    _dimensions = 3

    def __init__(self, fieldName, mesh, vectorFieldEntries, N=3):

        # Initialise volField
        super().__init__(fieldName, mesh, vectorFieldEntries, N)

    @property
    def dim(self) -> int:
        return self._dimensions

    # Write field in OpenFOAM format
    def write(self, timeValue):
        print(f"Writing {self._fieldName} field for time {timeValue} s")

        filePath = str(timeValue) + "/" + self._fieldName

        # Make new time directory
        if (os.path.exists(str(timeValue)) == False):
            os.mkdir(str(timeValue))

        # Write field in OF format
        with open(filePath, 'w') as file:
            # Write file banner
            file.write('/*\n' + BANNER + '*/\n')
            file.write(self.foamFileDict('volVectorField', self._fieldName, timeValue))
            file.write("\ndimensions      [0 1 0 0 0 0 0]; \n\n")

            # Write internal field
            file.write("internalField   nonuniform List<vector>\n")
            file.write(f"{self._mesh.nCells}\n(\n")

            for i in range(self._mesh.nCells):
                file.write(f'({" ".join(str(x) for x in self._cellValues[i])}) \n')
            file.write(")\n;\n\n")

            # Write boundary field
            file.write('boundaryField\n{\n')
            for patch in self._boundaryConditionsDict:
                patchType = self._boundaryConditionsDict[patch]['type']
                file.write(f'\t{patch}\n\t{{\n\t\ttype\t{patchType};\n')

                boundary = self._mesh._boundary
                nInternalFaces = self._mesh.nInternalFaces
                startFace = boundary[patch]['startFace']
                nFaces = boundary[patch]['nFaces']

                if patchType == 'fixedValue':
                    value = self._boundaryConditionsDict[patch]['value']['uniform']
                    file.write(f'\t\tvalue\tuniform ({" ".join(str(x) for x in value)});\n')

                if patchType == 'solidTraction':
                    file.write(f'\t\tvalue\tnonuniform List<vector>\n{nFaces}\n(\n')
                    for i in range(nFaces):
                        faceI = startFace + i - nInternalFaces
                        file.write(f'({" ".join(str(x) for x in self._boundaryValues[faceI])}) \n')
                    file.write(")\n;\n")

                file.write(f'\t}}\n')

            file.write('}\n')