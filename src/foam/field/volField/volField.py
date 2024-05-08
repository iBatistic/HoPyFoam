"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Volume field class as base clas for volume field
"""

import re
import numpy as np

from src.foam.field import field
from src.foam.foamFileParser import *
from src.foam.field.volField import *
from src.foam.field import localRegressionEstimator
from src.foam.field.volField.volFieldBoundaryConditions import *
from src.foam.decorators import timed

class volField(field, volFieldBoundaryConditions):

    def __init__(self, fieldName, mesh, boundaryAndInitialConditions):
        super().__init__(fieldName)

        # Initialise class variables
        self._mesh = mesh

        # Unpack boundaryAndInitialConditions tuple
        initialValue, self._dataType, self._boundaryConditionsDict \
            = boundaryAndInitialConditions

        self._initialValue = convert_to_float(initialValue)

        # Initialise volume field at cell centres and at boundary
        self._cellValues = self.initCellValues(mesh, self._initialValue, self._dimensions)
        self._boundaryValues = self.initBoundaryValues(mesh, self._boundaryConditionsDict)

        # Make interpolation molecule for faces
        self._facesInterpolationMolecule = self.makeFacesInterpolationMolecule()
        self._facesGaussPointsAndWeights = self.makeFacesGaussPointsAndWeights(self._boundaryConditionsDict, mesh)

        # Local Regression Estimator
        # Coefficients are calculated on object initialisation
        self._LRE = localRegressionEstimator(self, mesh)


    # TO_DO: Only works for scalar because initialValue is parsed as scalar, be careful with this later on!
    @classmethod
    def initCellValues(self, mesh, initialValue, dimensions):
        nCells = mesh.nCells()
        # Initialise cell values to 1 and multiply then with corresponding initialValue
        cellValues = np.ones((dimensions, nCells), dtype=float)

        # Loop over components and multiply each component with initial value
        for cmpt in range(dimensions):
            cellValues[cmpt] *= initialValue

        return cellValues

    # TO_DO: extend to account for vector, now is written for scalar values
    @classmethod
    def initBoundaryValues(self, mesh, boundaryConditionsDict):
        dimensions = self._dimensions
        boundaryFaceNb = mesh.boundaryFaceNb()

        # Initialise boundary values
        boundaryValues = \
            np.zeros((dimensions, boundaryFaceNb), dtype=float)

        # Loop over patches and call corresponding patch evaluate function
        for patch in boundaryConditionsDict:
            patchName = str(patch)
            patchType = boundaryConditionsDict[patchName]['type']
            if(patchType != 'empty'):
                patchValue = boundaryConditionsDict[patchName]['value']['uniform']

            evaluate = eval("self.evaluate." + patchType)

            evaluate(mesh, patchName, patchValue, boundaryValues, dimensions)

        return boundaryValues

    #TO_DO make it as class method
    def makeFacesInterpolationMolecule(self) -> list[int]:
        print(f"Calculating interpolation stencil for field {self._fieldName}")

        # Number of cells in stencil
        Nn = self._Nn
        facesMolecule = []

        for faceI in range(len(self._mesh._Cf)):

            internalFacesNb = self._mesh.nInternalFaces()

            # Construct interpolation molecule for internal and boudnary face
            # Loop over control volumes and take closest Nn neighbours
            distances = []
            faceCf = self._mesh._Cf[faceI]

            for cellI in range(self._mesh.nCells()):
                distance = np.linalg.norm(self._mesh.C()[cellI] - faceCf)
                distances.append((cellI, distance))

            # Sort faces according to distances and take Nn faces in stencil
            distancesSorted = \
                sorted(distances, key=lambda x: np.linalg.norm(x[1]))

            if(faceI < internalFacesNb):
                numberOfNeighbors = Nn
            else:
                # Stencil size is Nn - 1 because one of the cells of the stencil
                # is the face itself.
                numberOfNeighbors = Nn - 1

            faceMolecule = \
                [element[0] for element in distancesSorted[:numberOfNeighbors]]

            facesMolecule.append(faceMolecule)

        return facesMolecule

    @timed
    def makeFacesGaussPointsAndWeights(self, boundaryConditionsDict, mesh) -> list[list[np.ndarray]]:

        print(f"Calculating Gauss integration points for field {self._fieldName}")

        # Each face has list of corresponding Gauss points and their weights
        facesGaussPointsAndWeights = []

        # Loop over patches and check is there empty patches
        # If there are empty patches, Gauss points are calculated in one plane
        emptyDir = None
        for patch in boundaryConditionsDict:
            patchType = boundaryConditionsDict[patch]['type']
            if (patchType == 'empty'):
                print(f"Empty patch found, Gauss points are calculated in X-Y plane")
                # Z vector is empty direction
                emptyDir = 2
                break

        # Loop over all faces
        for faceI in range(len(mesh._Cf)):
            face = mesh.faces()[faceI]
            GaussPoints, weights = face.GaussPointsAndWeights(self._GaussPointsNb, emptyDir)
            facesGaussPointsAndWeights.insert(faceI,[weights, GaussPoints])

        return facesGaussPointsAndWeights

    def LRE(self) -> localRegressionEstimator:
        return self._LRE