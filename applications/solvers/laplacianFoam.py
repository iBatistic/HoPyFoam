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

# Execution start time, used to measured elapsed clock time
exec_start_time = timeModule.perf_counter()

# Get command line arguments
args = arg_parser().parse_args()

# Read and construct mesh, construct interpolation stencil
mesh = fvMesh()
solControl = solutionControl()

# Create field
print('Reading field T \n')

# Initialise scalar field T
T = volScalarField("T", mesh, readBoundaryAndInitialConditions("T"))

while(solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    '''
        Assemble and solve system of equations
    '''
    # Assemble the Laplacian matrix contributions
    #laplacianMatrix = fvm.construct(T, 'laplacian', laplacianScheme, [DT])

    # The actual matrix to solve for
    #matrix = ddtMatrix - laplacianMatrix

    #matrix.solve(fvSolution_T)

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')