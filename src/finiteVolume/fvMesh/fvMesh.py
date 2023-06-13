"""
 _____     _____     _____ _____ _____      |  Python code for High-order FVM
|  |  |___|  _  |_ _|   __|  |  |     |     |  Python Version: 3.10
|     | . |   __| | |   __|  |  | | | |     |  Code Version: 0.0
|__|__|___|__|  |_  |__|   \___/|_|_|_|     |  License: GPLv3
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


class polyMesh:

    def __init__(self):
        self._points = foamFileParser.read_points_file()
        self._faces = foamFileParser.read_faces_file()
        self._owner = foamFileParser.read_owner_file()
        self._neigbour = foamFileParser.read_neighbour_file()
        self._boundary = foamFileParser.read_boundary_File()


class fvMesh(polyMesh):

    def __init__(self):
        super().__init__()
        self._Cf, self._magSf, self._Sf = self._make_faces_data()
        self._V, self._C = self._make_cells_data()
        _meshSize = self._mesh_size()

    def _make_faces_data(self):

        Cf = np.zeros((np.size(self._faces), 3), float)
        Sf = np.zeros((np.size(self._faces), 3), float)
        magSf = np.zeros(np.size(self._faces), float)

        for i in range(np.size(self._faces, axis=0)):
            facePoints = np.empty((0, 3))
            for j in range(len(self._faces[i])):
                point = np.array(self._points[self._faces[i][j]])
                facePoints = np.append(facePoints, [point], axis=0)

            # Construct face using facePoints
            faceI = face(facePoints)
            Cf[i] = faceI.centre()
            Sf[i] = faceI.normal()
            magSf[i] = faceI.mag()

        return Cf, magSf, Sf

    def _make_cells_data(self) -> list:
        return [], []

    def _mesh_size(self):
        # return len(self._C)
        pass
