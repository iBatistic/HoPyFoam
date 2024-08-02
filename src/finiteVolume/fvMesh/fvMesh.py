"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Mesh data needed to do the Finite Volume discretisation.
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['fvMesh']

import numpy as np
import sys

import src.foam.foamFileParser as foamFileParser
from src.foam.primitives import face
from src.foam.primitives import tetrahedron
from src.foam.decorators import timed

class polyMesh:

    def __init__(self):
        self._points = foamFileParser.read_points_file()
        self.__faces = foamFileParser.read_faces_file()
        self._owner = foamFileParser.read_owner_file()
        self._neighbour = foamFileParser.read_neighbour_file()
        self._boundary = foamFileParser.read_boundary_File()

class fvMesh(polyMesh):

    def __init__(self):
        print("Creating mesh\n")
        super().__init__()
        self.__faces = self._makeFaces()
        self._Cf, self._magSf, self._Sf = self._makeFacesData()
        self._V, self._C = self._makeCellsData()
        self._meshSize = self.nCells

        # List of boundary cells
        self._boundaryCells = self.makeBoundaryCells()

        # List of cell faces
        self._cellFaces = self.makeCellFaces()

        # List of empty faces
        self._emptyFaces = self.makeEmptyFaces()

    @property
    def boundary(self):
        return self._boundary

    @property
    def cellFaces(self):
        return self._cellFaces

    @property
    def emptyFaces(self):
        return self._emptyFaces

    @property
    def boundaryCells(self):
        return self._boundaryCells

    # Number of faces
    @property
    def nFaces(self) -> int:
        return np.size(self._polyMesh__faces, axis=0)

    @property
    def magSf(self) -> np.ndarray:
        return self._magSf

    @property
    def Cf(self) -> np.ndarray:
        return self._Cf

    @property
    def V(self) -> np.ndarray:
        return self._V

    @property
    def nf(self) -> np.ndarray:
        return self._Sf/self._magSf

    @property
    def faces(self) -> face:
        return self.__faces

    @property
    def owners(self) -> np.ndarray:
        return self._owner

    @property
    def C(self) -> np.ndarray:
        return self._C

    @property
    def twoD(self) -> bool:
        for patch in self._boundary:
            if self._boundary[patch]['type'] == 'empty':
                return True
        return False

    @property
    def nInternalFaces(self) -> int:
        return self.nFaces-self.boundaryFaceNb

    @property
    def boundaryFaceNb(self) -> int:
        return np.size(self._owner, axis=0) - np.size(self._neighbour, axis=0)

    # Number of control volumes
    @property
    def nCells(self) -> int:
        return self._owner.max() + 1

    def facePatchType(self, faceI, volField, errorOnInternalFace=True) -> str:
        if faceI < self.nInternalFaces-1 and errorOnInternalFace:
            raise IndexError(
                "Face index {} corresponds to internal face".format(faceI))

        boundary = self._boundary
        for patch in boundary:
            startFace = boundary[patch]['startFace']
            nFaces = boundary[patch]['nFaces']
            if(faceI >= startFace and faceI < nFaces+startFace):
                #print(f'face {faceI} is on {patch} with start face {startFace} ')
                return volField._boundaryConditionsDict[patch]['type']

        # Face is internal
        return None

    def facePatchName(self, faceI, volField) -> str:
        if(faceI < self.nInternalFaces-1):
            raise IndexError(
                "Face index {} corresponds to internal face".format(faceI))

        boundary = self._boundary
        for patch in boundary:
            startFace = boundary[patch]['startFace']
            nFaces = boundary[patch]['nFaces']
            if(faceI >= startFace and faceI < nFaces+startFace):
                return patch

    @timed
    def _makeFaces(self):

        self.__faces = np.empty(self.nFaces, dtype=face)

        for i in range(self.nFaces):
            facePoints = np.empty((0, 3))
            for j in range(len(self._polyMesh__faces[i])):
                point = np.array(self._points[self._polyMesh__faces[i][j]])
                facePoints = np.append(facePoints, [point], axis=0)

            self.__faces[i] = face(facePoints)

        return self.__faces

    @timed
    def _makeFacesData(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        Cf = np.zeros((self.nFaces, 3), float)
        Sf = np.zeros((self.nFaces, 3), float)
        magSf = np.zeros((self.nFaces, 1), float)

        for i in range(self.nFaces):
            faceI = self.__faces[i]
            Cf[i] = faceI.centre()
            Sf[i] = faceI.normal()
            magSf[i] = faceI.mag()

        return Cf, magSf, Sf


    # TODO: Cell is not decoposed into
    @timed
    def _makeCellsData(self) -> tuple[np.ndarray, np.ndarray]:

        V = np.zeros(self.nCells, float)
        C = np.zeros((self.nCells, 3), float)

        nCellFaces = np.zeros(self.nCells, int)
        Cestimated = np.zeros((self.nCells, 3), float)

        # Estimate cell centre by simple average of cell face centres
        for facei in range(self._owner.size):
            Cestimated[self._owner[facei]] += self._Cf[facei]
            nCellFaces[self._owner[facei]] += 1

        for facei in range(self. _neighbour.size):
            Cestimated[self._neighbour[facei]] += self._Cf[facei]
            nCellFaces[self._neighbour[facei]] += 1

        for celli in range(np.size(Cestimated, axis=0)):
            Cestimated[celli] /= nCellFaces[celli]

        # Loop over each cell face and decomposing it into tetrahedrons by
        # using above estimated cell centre. Tetrahedrons volume and centre is
        # used to correct estimated centre and to calculate cell volume

        for facei in range(self._owner.size):
            f = self.__faces[facei]

            # Triangular face, no decomposition
            if(f.size() == 3):

                tetPoints = np.array([
                                [f.points()[2]],
                                [f.points()[1]],
                                [f.points()[0]],
                                [Cestimated[self._owner[facei]]]]
                              )

                tet = tetrahedron(tetPoints)

                tetVolume = tet.mag()
                C[self._owner[facei]] += tetVolume * tet.centre()
                V[self._owner[facei]] += tetVolume

            # Polygonal face
            else:
                faceCentre = f.centre()
                for pointi in range(f.size()):

                    tetPoints = np.array([
                                        [f.points()[pointi]],
                                        [f.prev_point(pointi)],
                                        [faceCentre],
                                        [Cestimated[self._owner[facei]]]])

                    tet = tetrahedron(tetPoints)

                    tetVolume = tet.mag()
                    C[self._owner[facei]] += tetVolume * tet.centre()
                    V[self._owner[facei]] += tetVolume

        for facei in range(self._neighbour.size):
            f = self.__faces[facei]

            # Triangular face, no decomposition
            if(f.size() == 3):
                tetPoints = np.array([
                    [f.points()[0]],
                    [f.points()[1]],
                    [f.points()[2]],
                    [Cestimated[self._neighbour[facei]]]]
                )

                tet = tetrahedron(tetPoints)

                tetVolume = tet.mag()
                C[self._neighbour[facei]] += tetVolume * tet.centre()
                V[self._neighbour[facei]] += tetVolume

            # Polygonal face
            else:
                faceCentre = f.centre()
                for pointi in range(f.size()):

                    tetPoints = np.array([
                        [f.points()[pointi]],
                        [f.next_point(pointi)],
                        [faceCentre],
                        [Cestimated[self._neighbour[facei]]]])

                    tet = tetrahedron(tetPoints)

                    tetVolume = tet.mag()
                    C[self._neighbour[facei]] += tetVolume * tet.centre()
                    V[self._neighbour[facei]] += tetVolume


        C /= V[:, np.newaxis]
        return V, C

    def makeBoundaryCells(self) -> list[int]:

        # Number of empty boundary faces
        boundary = self._boundary

        # Initialise list of boundary cells
        boundaryCells = []

        for patch in boundary:
            if boundary[patch]['type'] != 'empty':
                startFace = self.boundary[patch]['startFace']
                nFaces = self.boundary[patch]['nFaces']

                for faceI in range(startFace, startFace + nFaces):
                    bCellIndex = self._owner[faceI]
                    if bCellIndex not in boundaryCells:
                        boundaryCells.append(bCellIndex)

        return boundaryCells

    def makeCellFaces(self) -> list[list[int]]:

        cellFaces = [[] for _ in range(self.nCells)]

        for i in range(len(self._neighbour)):
            cellFaces[self._neighbour[i]].append(i)

        for i in range(len(self._owner)):
            cellFaces[self._owner[i]].append(i)

        return cellFaces

    def makeEmptyFaces(self) -> list[int]:
        emptyFaces = []

        boundary = self._boundary
        for patch in boundary:
            if boundary[patch]['type'] == 'empty':
                startFace = boundary[patch]['startFace']
                nFaces = boundary[patch]['nFaces']

                for faceI in range(nFaces):
                    emptyFaces.append(startFace+faceI)
        return emptyFaces