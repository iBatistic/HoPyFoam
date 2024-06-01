"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Compare solution with analytical from
    I do like CFD, VOL.1, Katate Masatsuka, edition II
    page 222, solution c.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.field.volField.volScalarField import volScalarField

mesh = fvMesh()

# Initialise scalar field T
T = volScalarField("T", mesh, readBoundaryAndInitialConditions('T', '1.0'))

# # Relative error, L_2 and L_infinity norm
avgRelError = 0.0
maxRelError = 0.0
L2 = 0.0
Linf = 0.0

# Get max value of solution, to be used for relative error
maxSolVal = 0.0
for cellI in range(mesh.nCells):
    x = mesh.C[cellI][0]
    y = mesh.C[cellI][1]

    #sol = x * x - y * y
    sol = abs(np.sin(5*y) * np.exp(5*x))
    if sol > maxSolVal:
        maxSolVal = sol

for cellI in range(mesh.nCells):

    x = mesh.C[cellI][0]
    y = mesh.C[cellI][1]

    #analySol = x*x - y*y + 1e-10
    analySol = np.sin(5*y) * np.exp(5*x)

    diff = abs(T._cellValues[cellI] - analySol)[0]
    #print(diff, analySol, x, y)

    relError = 100 * abs(diff / maxSolVal)
    avgRelError += relError

    if(relError > maxRelError):
       maxRelError = relError

    L2 += np.math.pow(diff, 2)

    if(diff > Linf):
        Linf = diff

L2 = np.math.sqrt(L2/mesh.nCells)
avgRelError /= mesh.nCells

print(f'\nAverage relative error: {avgRelError:.4e} %')
print(f'Maximal relative error: {maxRelError:.4e} %')
print(f'L_2 error: {L2:.4e}')
print(f'L_infinty error: {Linf:.4e}')