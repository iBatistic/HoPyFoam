"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Compare solution with analytical. Analytical solution is taken from:
    "Method of manufactured solutions code verification of elastostatic solid
     mechanics problems in a commercial finite element solver"
     Kenneth I. Aycock, Nuno Rebelo, Brent A. Craven, 
     Computers and Structures, 2020
"""

import sys
import os
import math

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.field.volField.volVectorField import volVectorField

mesh = fvMesh()

# Read vector field U
U = volVectorField("U", mesh, readVectorField('U', '1.0'))

diffSquared = 0.0
squaredAnalytical = 0.0
volume = 0.0

for cellI in range(mesh.nCells):

    x = mesh.C[cellI][0]
    y = mesh.C[cellI][1]
    
    C = 0.01
    n = 2
	
    analyticalUx = C * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)
    analyticalUy = C * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)

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
file.write(f"{mesh.nCells}    {relError}    {absError}\n")
file.close()
