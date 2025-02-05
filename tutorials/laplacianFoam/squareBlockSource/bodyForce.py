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

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.foamFileParser import *

mesh = fvMesh()

# Constant heat source
S = -1000

fileName = "bodyForce.dat"

# Remove old file
if os.path.exists(fileName):
    os.remove(fileName)

file = open(fileName, 'w')

# Loop over all control volumes
for celli in range(mesh.nCells):
    file.write(f"{S}\n")

file.close()
