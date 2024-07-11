"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Implementation of the Gaussian quadrature integration

    Source:
    D. A. Dunavant. High degree efficient symmetrical Gaussian quadrature rules
    for the triangle. International Journal for Numerical Methods in Engineering,
    21(6):1129–1148, jun 1985.
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['GaussianQuadrature']

import sys
import warnings
import numpy as np

# Dunavant quadrature points and weights for different orders of precision
TRIANGLE = \
{
    1:
        {
            'points': np.array([[0.333333333333333, 0.333333333333333, 0.333333333333333]]),
            'weights': np.array([1])
        },
    2:
        {
            'points': np.array([
                [0.666666666666667, 0.166666666666667, 0.166666666666667],
                [0.166666666666667, 0.666666666666667, 0.166666666666667],
                [0.166666666666667, 0.166666666666667, 0.666666666666667]
            ]),
            'weights': np.array([
                0.333333333333333,
                0.333333333333333,
                0.333333333333333
            ])
        },
    3:
        {
            'points': np.array([
                [0.333333333333333, 0.333333333333333, 0.333333333333333],
                [0.600000000000000, 0.200000000000000, 0.200000000000000],
                [0.200000000000000, 0.600000000000000, 0.200000000000000],
                [0.200000000000000, 0.200000000000000, 0.600000000000000]
            ]),
            'weights': np.array([
                -0.5625,
                0.520833333333333,
                0.520833333333333,
                0.520833333333333
            ])
        },
    4:
        {
            'points': np.array([
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459]
            ]),
            'weights': np.array([
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322
            ])
        },
    5:
        {
            'points': np.array([
                [0.333333333333333, 0.333333333333333, 0.333333333333333],
                [0.059715871789770, 0.470142064105115, 0.470142064105115],
                [0.470142064105115, 0.059715871789770, 0.470142064105115],
                [0.470142064105115, 0.470142064105115, 0.059715871789770],
                [0.797426985353087, 0.101286507323456, 0.101286507323456],
                [0.101286507323456, 0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.101286507323456, 0.797426985353087]
            ]),
            'weights': np.array([
                0.225000000000000,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827
            ])
        },
    6:
        {
            'points': np.array([
                [0.501426509658179, 0.249286745170910, 0.249286745170910],
                [0.249286745170910, 0.501426509658179, 0.249286745170910],
                [0.249286745170910, 0.249286745170910, 0.501426509658179],
                [0.873821971016996, 0.063089014491502, 0.063089014491502],
                [0.063089014491502, 0.873821971016996, 0.063089014491502],
                [0.063089014491502, 0.063089014491502, 0.873821971016996],
                [0.053145049844817, 0.310352451033784, 0.636502499121399],
                [0.053145049844817, 0.636502499121399, 0.310352451033784],
                [0.310352451033784, 0.053145049844817, 0.636502499121399],
                [0.636502499121399, 0.053145049844817, 0.310352451033784],
                [0.310352451033784, 0.636502499121399, 0.053145049844817],
                [0.636502499121399, 0.310352451033784, 0.053145049844817],
            ]),
            'weights': np.array([
                0.116786275726379,
                0.116786275726379,
                0.116786275726379,
                0.050844906370207,
                0.050844906370207,
                0.050844906370207,
                0.082851075618374,
                0.082851075618374,
                0.082851075618374,
                0.082851075618374,
                0.082851075618374,
                0.082851075618374,
            ])
        }
}

class GaussianQuadrature():

    def __init__(self):
        pass

    def integrateTriangle(self, points, func, rule):

        GPoints = TRIANGLE[rule]['points']
        weights = TRIANGLE[rule]['weights']

        integral = np.array([0.0, 0.0, 0.0])
        for i in range(len(GPoints)):
            x = GPoints[i, 0] * points[0, 0] + GPoints[i, 1] * points[1, 0] + GPoints[i, 2] * points[2, 0]
            y = GPoints[i, 0] * points[0, 1] + GPoints[i, 1] * points[1, 1] + GPoints[i, 2] * points[2, 1]

            integral += weights[i] * func(x,y)

        return integral

    def integrateQuad(self, points, func, rule):
        warnings.warn("Quad integration not implemented")
        return np.array([0, 0, 0])

    @classmethod
    def integrate2D(self, points, func, rule, divide=False):
        if divide:
            # Divide polygonal face into triangles and perform integration
            # for each triangle
            pass
        else:
            if len(points) == 3:
                return self.integrateTriangle(self, points, func, rule)
            elif len(points) == 4:
                return self.integrateQuad(self, points, func, rule)
            else:
                warnings.warn("Polyhedral face integration not implemented")
                sys.exit(1)



