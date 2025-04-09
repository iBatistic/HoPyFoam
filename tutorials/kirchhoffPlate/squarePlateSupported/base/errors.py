"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Compare solution with analytical solution for the thin square
    plate with uniform transverse pressure.
    The solution is taken from:
    Timoshenko, S., & Woinowsky-Krieger, S. (1959). Theory of plates and shells.
    It is assumed that the origin is at the centre of the plate.
    The plate dimensions are a x b, thickness of the plate is h.
"""

import sys
import os
import math

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.field.volField.volScalarField import volScalarField

mesh = fvMesh()

# Read vector field U
w = volScalarField("w", mesh, readScalarField('w', '1.0'))

diffSquared = 0.0
squaredAnalytical = 0.0
volume = 0.0

h, _, _ = readSolidProperties()
E, nu = readMechanicalProperties(convertToLameCoeffs=False)
D = E*pow(h,3)/(12*(1-pow(nu,2)))
a = 10
b = 10
p = 1000
N = 50

for cellI in range(mesh.nCells):

    x = mesh.C[cellI][0]
    y = mesh.C[cellI][1]

    # Sum initialization
    sum = 0.0

    # Loop through the series
    for m in range(1, N * 2 + 1, 2):
        alphaM = (m * np.pi * b) / (2 * a)
        t2 = ((alphaM * np.tanh(alphaM) + 2.0) / (2.0 * np.cosh(alphaM))) * np.cosh(m * np.pi * y / a)
        t3 = (1.0 / (2.0 * np.cosh(alphaM))) * (m * np.pi * y / a) * np.sinh(m * np.pi * y / a)
        t = (np.power(-1, 0.5 * (m - 1)) / np.power(m, 5)) * np.cos(m * np.pi * x / a)
        sum += t * (1.0 - t2 + t3)

    analyticalW = ((4 * p * np.power(a, 4)) / (np.power(np.pi, 5) * D)) * sum

    cellV = mesh.V[cellI]
    volume += cellV

    analyticalW = abs(analyticalW)
    numericW = abs(w._cellValues[cellI]).item()

    diffSquared += np.square(analyticalW - numericW) * cellV
    squaredAnalytical += np.square(analyticalW) * cellV

relError = 100 * np.sqrt(diffSquared / squaredAnalytical)
absError = np.sqrt(diffSquared / volume)

print(f'\nAverage relative error: {relError:.6f} %')
print(f'\nAverage absolute error: {absError:.6f} %')

fileName = "errors.dat"
# Remove old file with results
if (os.path.exists(fileName)):
    os.remove(fileName)

# Average triangle lenght
A = a*b / mesh.nCells
l = np.sqrt(2*A)

file = open(fileName, 'w')
file.write(f"{mesh.nCells}    {l}    {relError}    {absError}\n")
file.close()
