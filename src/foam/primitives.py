"""
 _____     _____     _____ _____ _____      |  Python code for High-order FVM
|  |  |___|  _  |_ _|   __|  |  |     |     |  Python Version: 3.10
|     | . |   __| | |   __|  |  | | | |     |  Code Version: 0.0
|__|__|___|__|  |_  |__|   \___/|_|_|_|     |  License: GPLv3
                |___|

Description
    Primitives
"""

import numpy as np


class face():
    def __init__(self, facePoints):
        self._facePoints = facePoints

        # Check size of input array
        if np.size(self._facePoints, axis = 0) < 3:
            raise IndexError(
                "Face is initialised with {} points".format(len(facePoints)))

    # TODO: Centre point for polygonal faces is not the same as in face.C,
    #  lines 515-545. Probably this is less accurate method on non-flat faces.
    def centre(self):

        # Direct calculation for triangle
        if (np.size(self._facePoints, axis = 0) == 3):
            return (1.0 / 3.0) \
                * (
                    self._facePoints[0]
                  + self._facePoints[1]
                  + self._facePoints[2]
                )

        # Geometric centre decomposition
        nPoints = len(self._facePoints)
        centre = (1.0/nPoints) * np.sum(self._facePoints, axis=0)

        return centre

    def normal(self):

        nPoints = len(self._facePoints)

        #TODO: Here triangle class should be used
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
                p2 = self._facePoints[i+1]
            else:
                p2 = self._facePoints[0]

            vec1 = p2 - p1
            vec2 = faceCentre - p1
            #TODO: Here trinagle class should be used
            normal += 0.5 * (np.cross(vec1, vec2))

        return normal

    def mag(self):
        return np.linalg.norm(self.normal())
