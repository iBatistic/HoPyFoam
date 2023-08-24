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

import src.foam.foamFileParser as foamFileParser
from src.foam.primitives import face
from src.foam.primitives import tetrahedron
from src.foam.decorators import timed

class polyMesh:

    def __init__(self):
        self._points = foamFileParser.read_points_file()
        self.__faces = foamFileParser.read_faces_file()
        self._owner = foamFileParser.read_owner_file()
        self._neigbour = foamFileParser.read_neighbour_file()
        self._boundary = foamFileParser.read_boundary_File()


class fvMesh(polyMesh):

    def __init__(self):
        super().__init__()
        self.__faces = self._make_faces()
        self._Cf, self._magSf, self._Sf = self._make_faces_data()
        self._V, self._C = self._make_cells_data()
        self._meshSize = self.nb_cvs()

    # Number of faces
    def nb_faces(self) -> int:
        return np.size(self._polyMesh__faces, axis=0)

    # Number of control volumes
    def nb_cvs(self):
        return self._owner.max() + 1

    @timed
    def _make_faces(self):

        self.__faces = np.empty(self.nb_faces(), dtype=face)

        for i in range(self.nb_faces()):
            facePoints = np.empty((0, 3))
            for j in range(len(self._polyMesh__faces[i])):
                point = np.array(self._points[self._polyMesh__faces[i][j]])
                facePoints = np.append(facePoints, [point], axis=0)

            self.__faces[i] = face(facePoints)

        return self.__faces

    @timed
    def _make_faces_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        Cf = np.zeros((self.nb_faces(), 3), float)
        Sf = np.zeros((self.nb_faces(), 3), float)
        magSf = np.zeros(self.nb_faces(), float)

        for i in range(self.nb_faces()):
            faceI = self.__faces[i]
            Cf[i] = faceI.centre()
            Sf[i] = faceI.normal()
            magSf[i] = faceI.mag()

        return Cf, magSf, Sf

    # TODO: Cell is not decoposed into
    @timed
    def _make_cells_data(self) -> tuple[np.ndarray, np.ndarray]:

        V = np.zeros(self.nb_cvs(), float)
        C = np.zeros((self.nb_cvs(), 3), float)

        nCellFaces = np.zeros(self.nb_cvs(), int)
        Cestimated = np.zeros((self.nb_cvs(), 3), float)

        # Estimate cell centre by simple average of cell face centres
        for facei in range(self._owner.size):
            Cestimated[self._owner[facei]] += self._Cf[facei]
            nCellFaces[self._owner[facei]] += 1

        for facei in range(self. _neigbour.size):
            Cestimated[self._neigbour[facei]] += self._Cf[facei]
            nCellFaces[self._neigbour[facei]] += 1

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

        for facei in range(self._neigbour.size):
            f = self.__faces[facei]

            # Triangular face, no decomposition
            if(f.size() == 3):
                tetPoints = np.array([
                    [f.points()[0]],
                    [f.points()[1]],
                    [f.points()[2]],
                    [Cestimated[self._neigbour[facei]]]]
                )

                tet = tetrahedron(tetPoints)

                tetVolume = tet.mag()
                C[self._neigbour[facei]] += tetVolume * tet.centre()
                V[self._neigbour[facei]] += tetVolume

            # Polygonal face
            else:
                faceCentre = f.centre()
                for pointi in range(f.size()):

                    tetPoints = np.array([
                        [f.points()[pointi]],
                        [f.next_point(pointi)],
                        [faceCentre],
                        [Cestimated[self._neigbour[facei]]]])


                    tet = tetrahedron(tetPoints)

                    tetVolume = tet.mag()
                    C[self._neigbour[facei]] += tetVolume * tet.centre()
                    V[self._neigbour[facei]] += tetVolume


        C /= V[:, np.newaxis]
        return V, C