"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Solid solver with small strains and small rotations assumption (linear geometry).
    Total displacement is the primary unknown.
    The stress is calculated according to linear-elastic Hooke law.
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'

import time as timeModule
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.argList import arg_parser
from src.foam.foamFileParser import *
from src.finiteVolume.cfdTools import solutionControl
from src.foam.field.volField.volVectorField import volVectorField
from src.finiteVolume.fvMatrices.fvm.fvm import fvm

# Execution start time, used to measured elapsed clock time
exec_start_time = timeModule.perf_counter()

# Get command line arguments
args = arg_parser().parse_args()

# Read and construct mesh, construct interpolation stencil
mesh = fvMesh()

solControl = solutionControl()

# Initialise displacement vector field U
# N is interpolation order, Nn is number of cells in face stencil and GpNb
# is number of Gauss points per face
#U = volVectorField("U", mesh, readVectorField("U"),  N=3, Nn=20, GpNb=7)
#U = volVectorField("U", mesh, readVectorField("U"),  N=2, Nn=16, GpNb=7)
U = volVectorField("U", mesh, readVectorField("U"),  N=1, Nn=13, GpNb=7)

# Read mechanichalProperties dict to get first and second Lame parameters
mu, lam = readMechanicalProperties()

while (solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    # Assemble the Laplacian, LaplacianTranspose and LaplacianTrace matrices
    laplacian = fvm.construct(U, 'Laplacian', mu)
    laplacianTranspose = fvm.construct(U, 'LaplacianTranspose', mu)
    laplacianTrace = fvm.construct(U, 'LaplacianTrace', lam)
    bodyForce = fvm.construct(U, 'bodyForce')

    Matrix = laplacian + laplacianTranspose + laplacianTrace + bodyForce

    # Solve system matrix
    Matrix.solve()

    # Write results
    U.write(solControl.time())

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')