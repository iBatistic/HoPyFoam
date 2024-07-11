"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Compare solution with analytical. Analytical solution is taken from
    Demirdzic I., "A fourth-order finite volume method for structural analysis"
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.field.volField.volVectorField import volVectorField

mesh = fvMesh()

# Read vector field U
U = volVectorField("U", mesh, readVectorField('U', '1.0'))

F = -4
E = 30000
nu = 0.3
L = 50
b = 1
I = (2/3)*pow(b,3)

diffSquared = 0.0
squaredAnalytical = 0.0
volume = 0.0

for cellI in range(mesh.nCells):

    x = L - mesh.C[cellI][0]
    y = mesh.C[cellI][1] - 1

    analyticalUx = (F/(E*I)) * (-0.5*y*pow(x,2) + pow(y,3)*(2.0-nu)/6.0 - (1+nu)*pow(b,2)*y + 0.5*pow(L,2)*y )
    analyticalUy = (F/(E*I)) * (nu*0.5*x*pow(y,2) + pow(x,3)/6 - 0.5*x*pow(L,2) + (1/3)*pow(L,3))

    cellV = mesh.V[cellI]
    volume += cellV

    analyticalDispMag = np.linalg.norm([analyticalUx, analyticalUy, 0.0])
    dispMag = np.linalg.norm(U._cellValues[cellI])

    diffSquared += pow(analyticalDispMag - dispMag,2) * cellV
    squaredAnalytical += pow(analyticalDispMag,2) * cellV

relError = 100 * math.sqrt(diffSquared / squaredAnalytical)
absError = math.sqrt(diffSquared / volume)

print(f'\nAverage relative error: {relError:.6f} %')
print(f'\nAverage absolute error: {absError:.6f} %')

fileName = "errors.dat"
# Remove old file with results
if (os.path.exists(fileName)):
    os.remove(fileName)

file = open(fileName, 'w')
file.write(f"#nCells    relError    absError\n")
file.write(f"{mesh.nCells}    {relError}    {absError}\n")
file.close()
