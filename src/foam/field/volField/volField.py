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

    def __init__(self, fieldName, mesh, scalarFieldEntries):
        super().__init__(fieldName)

        # Initialise class variables
        self._mesh = mesh

        # Unpack scalarFieldEntries tuple
        cellValues, self._dataType, self._boundaryConditionsDict \
            = scalarFieldEntries

        # Set values at cell centres
        self._cellValues = self.setCellValues(mesh, cellValues, self._dimensions)

        # Make interpolation molecule for faces
        self._facesInterpolationMolecule = self.makeFacesInterpolationMolecule()
        self._facesGaussPointsAndWeights = self.makeFacesGaussPointsAndWeights(self._boundaryConditionsDict, mesh)

        # Local Regression Estimator
        # Coefficients are calculated on object initialisation
        self._LRE = localRegressionEstimator(self, mesh)


    def LRE(self) -> localRegressionEstimator:
        return self._LRE

    @classmethod
    def setCellValues(self, mesh, cellValues, dimensions):

        # Initialise cell values array
        values = np.zeros((dimensions, mesh.nCells), dtype=float)

        # Case of uniform internal field
        if (len(cellValues) == 1):
            values[:] = cellValues[0][0]
        else:
            # Case of non-uniform internal field
            for index in range(mesh.nCells):
                for cmpt in range(dimensions):
                    values[cmpt][index] = cellValues[index][cmpt]
        return values

    #This is brute force approach. Should be done more efficiently!
    def makeFacesInterpolationMolecule(self) -> list[int]:
        print(f"Calculating interpolation stencil for field {self._fieldName}")

        # Number of cells in stencil
        Nn = self._Nn
        facesMolecule = []

        for faceI in range(len(self._mesh._Cf)):

            internalFacesNb = self._mesh.nInternalFaces

            # Construct interpolation molecule for internal and boudnary face
            # Loop over control volumes and take closest Nn neighbours
            distances = []
            faceCf = self._mesh._Cf[faceI]

            for cellI in range(self._mesh.nCells):
                distance = np.linalg.norm(self._mesh.C[cellI] - faceCf)
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

        # If the mesh is 2D, Gauss points are calculated in one plane
        emptyDir = None
        if mesh.twoD:
            print(f"Empty patch found, Gauss points are calculated in X-Y plane")
            emptyDir = 2

        # Loop over all faces and calculate Gauss points and weights
        for faceI in range(len(mesh._Cf)):
            face = mesh.faces[faceI]
            GaussPoints, weights = face.GaussPointsAndWeights(self._GaussPointsNb, emptyDir)
            facesGaussPointsAndWeights.insert(faceI,[weights, GaussPoints])

        return facesGaussPointsAndWeights

