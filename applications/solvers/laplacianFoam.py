"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Steady-state Laplace's equation solver
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__  = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'

import time as timeModule
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.argList import arg_parser
from src.foam.foamFileParser import *
from src.finiteVolume.cfdTools import solutionControl
from src.foam.field.volField.volScalarField import volScalarField
from src.finiteVolume.fvMatrices.fvm.fvm import fvm

# Execution start time, used to measured elapsed clock time
exec_start_time = timeModule.perf_counter()

# Get command line arguments
args = arg_parser().parse_args()

# Read and construct mesh, construct interpolation stencil
mesh = fvMesh()
solControl = solutionControl()

# Initialise scalar field T, N is interpolation order
T = volScalarField("T", mesh, readScalarField("T"), N=1)

# Read diffusivity from transportProperties dict
gammaDimensions, gammaValue = readTransportProperties('DT')

while(solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    # Assemble the Laplacian matrix
    laplacian = fvm.construct(T, 'Laplacian', gammaValue)
    bodyForce = fvm.construct(T, 'bodyForce')

    Matrix = laplacian + bodyForce

    # Solve system matrix
    Matrix.solve()

    # Write results
    T.write(solControl.time())

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')