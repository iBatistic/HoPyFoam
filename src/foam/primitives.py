"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Different geometrical primitives
"""

import numpy as np

class face():
    def __init__(self, facePoints):
        self._facePoints = facePoints

        # Check size of input array
        nbFacePoints = np.size(facePoints, axis=0)
        if  nbFacePoints < 3:
            raise IndexError(
                "Face is initialised with {} points".format(nbFacePoints))

    # Number of face points
    def size(self) -> int:
        return np.size(self._facePoints, axis=0)

    # Return face points
    def points(self) -> np.ndarray:
        return self._facePoints

    def next_point(self, index):
        next_index = 0 if index == (self.size() - 1) else index + 1
        return self._facePoints[next_index]

    def prev_point(self, index):
        prev_index = index - 1 if index else self.size() - 1
        return self._facePoints[prev_index]

    # TODO: Centre point for polygonal faces is not the same as in face.C,
    #  lines 515-545. Probably this is less accurate method on non-flat faces.
    def centre(self):

        # Direct calculation for triangle
        if (np.size(self._facePoints, axis=0) == 3):
            return (1.0 / 3.0) \
                * (
                        self._facePoints[0]
                        + self._facePoints[1]
                        + self._facePoints[2]
                )

        # Geometric centre decomposition
        nPoints = len(self._facePoints)
        centre = (1.0 / nPoints) * np.sum(self._facePoints, axis=0)

        return centre

    def normal(self):

        nPoints = len(self._facePoints)

        # TODO: Here triangle class should be used
        # For triangle do the direct calculation
        if (nPoints == 3):
            vec1 = self._facePoints[1] - self._facePoints[0]
            vec2 = self._facePoints[2] - self._facePoints[0]
            return 0.5 * (np.cross(vec1, vec2))

        # For polygonal faces do central decomposition
        faceCentre = self.centre()
        normal = np.zeros(3)
        for i in range(nPoints):
            # First point
            p1 = self._facePoints[i]

            # Second point.
            if i < (nPoints - 1):
                p2 = self._facePoints[i + 1]
            else:
                p2 = self._facePoints[0]

            vec1 = p2 - p1
            vec2 = faceCentre - p1
            # TODO: Here trinagle class should be used
            normal += 0.5 * (np.cross(vec1, vec2))

        return normal

    def mag(self):
        return np.linalg.norm(self.normal())


class tetrahedron():

    def __init__(self, tetrahedronPoints):

        # Check size of input array
        nbTetPoints = np.size(tetrahedronPoints, axis=0)
        if nbTetPoints != 4:
            raise IndexError(
                "Tetrahedron is initialised with {} points".format(nbTetPoints))

        self._a = tetrahedronPoints[0]
        self._b = tetrahedronPoints[1]
        self._c = tetrahedronPoints[2]
        self._d = tetrahedronPoints[3]

    # Return tetrahedron volume
    def mag(self) -> float:
        vec1 = self._b - self._a
        vec2 = self._c - self._a
        vec3 = self._d - self._a
        return np.concatenate((1.0 / 6.0) * (np.cross(vec1, vec2) @ vec3.T))

    def centre(self) -> np.ndarray:
        return np.concatenate(0.25*(self._a + self._b + self._c + self._d))