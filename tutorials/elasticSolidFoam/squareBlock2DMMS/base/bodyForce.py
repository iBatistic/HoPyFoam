"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Calculate and write body force distribution according to MMS.
    Example is taken from: 
    "Method of manufactured solutions code verification of elastostatic solid
     mechanics problems in a commercial finite element solver"
     Kenneth I. Aycock, Nuno Rebelo, Brent A. Craven, 
     Computers and Structures, 2020
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.field.volField.volVectorField import volVectorField
from src.foam.field.GaussianQuadrature import GaussianQuadrature

mesh = fvMesh()

# Read vector field U
U = volVectorField("U", mesh, readVectorField('U', '0'))

# Read mechanichalProperties dict to get first and second Lame parameters
mu, lam = readMechanicalProperties()

def bodyForceFunc(x,y,z=0):

    C = 0.01
    n = 2
	
    d2Ux_DxDx = -C * pow(np.pi, 2) * pow(n, 2) * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)
    d2Ux_DyDy = -C * pow(np.pi, 2) * pow(n, 2) * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)
    d2Ux_DyDx = C * pow(np.pi, 2) * pow(n, 2) * np.cos(np.pi*n*x) * np.cos(np.pi*n*y)

    d2Uy_DxDx = -C * pow(np.pi, 2) * pow(n, 2) * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)
    d2Uy_DyDy = -C * pow(np.pi, 2) * pow(n, 2) * np.sin(np.pi*n*x) * np.sin(np.pi*n*y)
    d2Uy_DxDy = C * pow(np.pi, 2) * pow(n, 2) * np.cos(np.pi*n*x) * np.cos(np.pi*n*y)

    bx = (mu+lam) * (d2Ux_DxDx + d2Uy_DxDy) + mu * (d2Ux_DxDx + d2Ux_DyDy)
    by = (mu+lam) * (d2Ux_DyDx + d2Uy_DyDy) + mu * (d2Uy_DxDx + d2Uy_DyDy)
    bz = 0

    return np.array([bx, by, bz])


fileName = "bodyForce.dat"

# Remove old file
if (os.path.exists(fileName)):
    os.remove(fileName)

file = open(fileName, 'w')

# Loop over empty patch faces with z=1
for patch in mesh.boundary:
    if mesh.boundary[patch]['type'] == 'empty':
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']
        for i in range(nFaces):
            d = np.dot(mesh.nf[startFace + i], np.array([0, 0, 1]))
            if d > 0.999 and d < 1.001:
                # Face points
                points = mesh.faces[startFace + i].points()

                # Integrated face body force
                bodyForce = GaussianQuadrature.integrate2D(points, bodyForceFunc, 6)

                file.write(f"{bodyForce[0]}    {bodyForce[1]}    {bodyForce[2]}\n")
file.close()



