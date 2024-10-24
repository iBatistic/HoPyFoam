"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Surface vector field class
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['surfaceVectorField']

import os
import shutil
import warnings
import numpy as np

from src.foam.field.surfaceField import surfaceField
from src.foam.argList import BANNER
from src.foam.field.surfaceField.surfaceFieldBoundaryConditions import *

class surfaceVectorField(surfaceField):
    # Field dimensions
    _dimensions = 3

    def __init__(self, fieldName, baseField):
        # Initialise volField
        super().__init__(fieldName)

        self._baseField = baseField

        # Interpolate cell centre values to face Gauss points
        self._faceValues = self.interpolateCellToFace()

    def __getitem__(self, index):
        return self._faceValues[index]

    def evaluate(self):
        self._faceValues = self.interpolateCellToFace()

    def interpolateCellToFace(self):
        mesh = self._baseField._mesh
        psi = self._baseField

        # Initialize faces Gauss points values
        GaussPointsValues = np.zeros((mesh.nFaces, psi.Ng, psi.dim))

        # Interior faces
        for faceI in range(mesh.nInternalFaces):
            faceStencil = psi._facesInterpolationMolecule[faceI]
            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                # Gauss point interpolation coefficient vector
                c = psi.LRE().internalCoeffs[faceI][i]

                # Loop over Gauss point interpolation stencil
                for j, cellIndex in enumerate(faceStencil):
                    GaussPointsValues[faceI][i] += \
                        np.array(psi._cellValues[cellIndex]) * c[j]

        # Enforce boundary conditions at boundary faces
        for patch in psi._boundaryConditionsDict:
            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("surfaceFieldBoundaryConditions.evaluate." + patchType)
            patchContribution(self, psi, patch, GaussPointsValues)

        return GaussPointsValues

    def extractFaceCentreValues(self):
        """Input field is given for all Gauss points, this function extract
        values only at Gauss points located at face centre."""
        print(self._faceValues)
        values = np.zeros((self._baseField._mesh.nFaces, 3))

        if self._baseField._GaussPointsNb % 2 != 0:
            for faceI in range(len(self._faceValues)):
                values[faceI] = self._faceValues[faceI][self._baseField._GaussPointsNb//2]

        return values

    @property
    def dim(self) -> int:
        """Return the dimension of the vector field."""
        return self._dimensions

    def write(self, timeValue):
        """ Write field to disc in OpenFOAM format"""
        filePath = str(timeValue) + "/" + self._fieldName

        # Make new time directory
        if not os.path.exists(str(timeValue)):
            os.mkdir(str(timeValue))

        # Extract values at face centre to be able to match OF format when writing
        faceCentredValues = self.extractFaceCentreValues()

        # Write field in OF format
        with open(filePath, 'w') as file:
            # Write file banner
            file.write('/*\n' + BANNER + '*/\n')
            file.write(self.foamFileDict('surfaceVectorField', self._fieldName, timeValue))
            file.write("\ndimensions      [0 1 0 0 0 0 0]; \n\n")

            # Write internal field
            file.write("internalField   nonuniform List<vector>\n")
            file.write(f"{self._baseField._mesh.nInternalFaces}\n(\n")

            for i in range(self._baseField._mesh.nInternalFaces):
                file.write(f'({" ".join(str(x) for x in faceCentredValues[i])})\n')
            file.write(")\n;\n\n")

            # Write boundary field
            file.write('boundaryField\n{\n')
            for patch in self._baseField._boundaryConditionsDict:

                if self._baseField._boundaryConditionsDict[patch]['type'] == 'empty':
                    file.write(f'\t{patch}\n\t{{\n\t\ttype\tempty;\n')
                    file.write(f'\t}}\n')
                    continue

                file.write(f'\t{patch}\n\t{{\n\t\ttype\tcalculated;\n')

                boundary = self._baseField._mesh._boundary
                startFace = boundary[patch]['startFace']
                nFaces = boundary[patch]['nFaces']

                file.write(f'\t\tvalue\tnonuniform List<vector>\n{nFaces}\n(\n')

                for faceI in range(startFace, startFace + nFaces):
                    file.write(f'({" ".join(str(x) for x in faceCentredValues[faceI])})\n')
                file.write(")\n;\n")

                file.write(f'\t}}\n')

            file.write('}\n')