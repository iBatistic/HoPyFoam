"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Assign a constant heat source term for all control volumes in the domain.
"""

import sys
import os
import math

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *
from src.foam.argList import arg_parser

# Write header
args = arg_parser().parse_args()

print(f'Writing pressure field in bodyForce.dat to be used as source field\n')

mesh = fvMesh()

# Constant pressure
p = -1000

# Solver read source only if this file name is used
fileName = "bodyForce.dat"

# Remove old file
if os.path.exists(fileName):
    os.remove(fileName)

file = open(fileName, 'w')

# Loop over all control volumes
for cellI in range(mesh.nCells):

    # Cell volume
    V = mesh.V[cellI]

    # Body force multyply pressure with cell volume, so we need to divide it
    # here to have the right pressure value in the source
    source = p #/ V
    file.write(f"{source}\n")

file.close()

print(f'End\n')
