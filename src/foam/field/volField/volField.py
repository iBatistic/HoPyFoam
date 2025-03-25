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
import warnings

from src.foam.field import field
from src.foam.foamFileParser import *
from src.foam.field.volField import *
from src.foam.field import localRegressionEstimator

from src.foam.field.volField.volFieldBoundaryConditions import *
from src.foam.decorators import timed

class volField(field, volFieldBoundaryConditions):

    def __init__(self, fieldName, mesh, fieldEntries, N, cellLRE=False):

        super().__init__(fieldName)

        # Order of interpolation (1 = linear, 2 = quadratic, 3 = qubic...)
        self._N = N

        # Number of terms in Taylor expansion
        # Example: Second order (quadratic interpolation) has 6 terms in 2D case
        # Example: First order (linear interpolation) has 3 terms in 2D case
        self. _Np  = self.TaylorTermsNumber(self._N, mesh.twoD)

        # Number of cells in interpolation stencil
        self._Nn = self.stencilSize(mesh.twoD, N)

        # Number of integration points per CV face
        self._GaussPointsNb = self.integrationPointsNb(mesh.twoD, N)

        print(f'Field {fieldName}, N: {self._N}, Np: {self._Np}, Nn: {self.Nn},  Ng: {self._GaussPointsNb}')

        # A quick check for stencil size requirement
        if (self._Np >= self._Nn):
            warnings.warn(f"\nNumber of neighbours {self._Nn} is smaller than number "
                          f"of Taylor expansion terms {self._Np}\n", stacklevel=3)

        # Initialise class variables
        self._mesh = mesh

        # Unpack fieldEntries tuple
        cellValuesData, self._dataType, self._boundaryConditionsDict = fieldEntries

        # Set values at cell centres
        self._cellValues = self.initCellValues(mesh, cellValuesData, self.dim)
        self._boundaryValues = self.initBoundaryValues(mesh, self.dim)

        # Store previous iteration cell values, used for residual calculation and under-relaxation
        self._prevIter =  self._cellValues

        # Make interpolation molecule for faces and cell centres
        print(f"Calculating interpolation stencil for field {self._fieldName}")
        self._facesInterpolationMolecule = self.makeFacesInterpolationMolecule()
        self._cellsInterpolationMolecule = self.makeCellsInterpolationMolecule()

        # Make Gauss points on faces and corresponding weights
        print(f"Calculating Gauss integration points for field {self._fieldName}")
        self._facesGaussPointsAndWeights = self.makeFacesGaussPointsAndWeights(self._boundaryConditionsDict, mesh)

        # Local Regression Estimator
        # Coefficients are calculated on object initialisation
        self._LRE = localRegressionEstimator(self, mesh, cellLRE)

    def LRE(self) -> localRegressionEstimator:
        return self._LRE

    # Return previous iteration field (only cell values)
    @property
    def prevIter(self) -> np.ndarray:
        return self._prevIter

    # Return number of terms in Taylor expression
    # For two-dimensional cases we are ignoring terms related to z coordinate!
    @property
    def Np(self) -> int:
        return self._Np

    @property
    def Ng(self) -> int:
        return self._GaussPointsNb

    # Return number of neighbours for this field type
    @property
    def Nn(self) -> int:
        return self._Nn

    # Interpolation  order, polynomial degree
    @property
    def N(self) -> int:
        return self._N

    # Returns number of terms in Taylor expression, hard-coded
    @classmethod
    def TaylorTermsNumber(self, N, twoD) -> int:
        """
        Returns the number of terms in Taylor polynomial of order N.
        """
        TaylorTerms2D = {1: 3, 2: 6, 3: 10}

        if twoD:
            if N in TaylorTerms2D:
                return TaylorTerms2D[N]
            else:
                raise ValueError(f'Taylor polynom {N} not implemented')
        else:
            raise ValueError("Only 2D implemented")

    @classmethod
    def stencilSize(self, twoD: bool, N: int) -> int:
        """
        Returns the stencil size based on the dimensionality and the stencil
        order.
        """
        stencilSize2D = {1: 14, 2: 16, 3: 20}

        if twoD:
            if N in stencilSize2D:
                return stencilSize2D[N]
            else:
                raise ValueError(f'Stencil size not implemented for {N} '
                                 'Taylor interpolation polynom')
        else:
            raise ValueError("Only 2D stencils are implemented")

    @classmethod
    def integrationPointsNb(self, twoD, Np) -> int:
        """
        Returns the number of Gauss points per CV face.
        """
        if twoD:
            return 7
        else:
            raise ValueError("Only 2D is implemented")

    def evaluateBoundary(self):

        bdDict = self._boundaryConditionsDict

        # Loop over patches and call corresponding patch evaluate function
        for patch in bdDict:

            patchName = str(patch)
            patchType = bdDict[patchName]['type']

            evaluate = eval("volFieldBoundaryConditions.evaluate." + patchType)

            evaluate(self, self._mesh, bdDict,  patchName, self._boundaryValues)

    # Convert cell data from 0/field to cellValues list
    # This data can be uniform or non-uniform
    @classmethod
    def initCellValues(self, mesh, cellValuesData, dimensions):

        # Initialise cell values array
        values = np.zeros((mesh.nCells, dimensions), dtype=float)

        # Case of uniform internal field
        if (len(cellValuesData) == 1):
            values[:] = cellValuesData[0]

        else:
            # Case of non-uniform internal field
            for index in range(mesh.nCells):
                values[index] = cellValuesData[index]
        return values

    @classmethod
    def initBoundaryValues(self, mesh, dimensions):

        # Initialise boundary array
        boundaryValues = \
            np.zeros((mesh.boundaryFaceNb, dimensions), dtype=float)

        return boundaryValues

    def correct(self, other, relaxation):
        for cellI in range(self._mesh.nCells):
            self._cellValues[cellI] += np.array(other._cellValues[cellI]) * relaxation

    #This is brute force approach. Should be done more efficiently!
    def makeFacesInterpolationMolecule(self) -> list[int]:

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

    def makeCellsInterpolationMolecule(self) -> list[int]:

        cellsMolecule = []

        mesh = self._mesh
        Nn = self._Nn

        for cellI in range(mesh.nCells):
            # Construct interpolation molecule for cell centre
            distances = []
            cellC = self._mesh._C[cellI]

            for cellN in range(mesh.nCells):
                distance = np.linalg.norm(mesh.C[cellN] - cellC)
                distances.append((cellN, distance))

            # Sort faces according to distances and take Nn faces in stencil
            distancesSorted = \
                sorted(distances, key=lambda x: np.linalg.norm(x[1]))

            if cellI not in mesh.boundaryCells:
                numberOfNeighbors = Nn
            else:
                # Reduce the number of neighbors in stencil for the number of
                # ghost boundary faces.
                # For example, some boundary cells have 2 boundary faces
                boundaryFaces = 0
                for faceI in mesh.cellFaces[cellI]:
                    if faceI >= mesh.nInternalFaces:
                        if faceI not in mesh.emptyFaces:
                            boundaryFaces += 1

                numberOfNeighbors = Nn - boundaryFaces

            cellMolecule = \
                [element[0] for element in distancesSorted[:numberOfNeighbors]]

            cellsMolecule.append(cellMolecule)

        return cellsMolecule


    @timed
    def makeFacesGaussPointsAndWeights(self, boundaryConditionsDict, mesh) -> list[list[np.ndarray]]:

        # Each face has list of corresponding Gauss points and their weights
        facesGaussPointsAndWeights = []

        # If the mesh is 2D, Gauss points are calculated in one plane
        emptyDir = None
        if mesh.twoD:
            print(f"Empty patch found, Gauss points are calculated in"
                  f" X-Y plane\n")

            emptyDir = 2

        # Loop over all faces and calculate Gauss points and weights
        for faceI in range(len(mesh._Cf)):
            face = mesh.faces[faceI]
            GaussPoints, weights = face.GaussPointsAndWeights(self._GaussPointsNb, emptyDir)
            facesGaussPointsAndWeights.insert(faceI,[weights, GaussPoints])

        return facesGaussPointsAndWeights

