"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Based on "rotation-free formulation" of Torlak (2006).
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.unizg.hr, philip.cardiff@ucd.ie'

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

# Initialise deflection scalar field w, N is interpolation order
w = volScalarField("w", mesh, readScalarField("w"),  N=3)

# Initialise moment sum  scalar field M, N is interpolation order
M = volScalarField("M", mesh, readScalarField("M"),  N=3)

# Read mechanichalProperties dict to get Young modulus and Poisson's ratio
E, nu = readMechanicalProperties(convertToLameCoeffs=False)

h=0.1
bendingStiffness = E*pow(h,3)/(12*(1-pow(nu,2)))

while (solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    iCorr = 0
    while True:
        # Assemble the momentum sum matrix
        laplacianM = fvm.construct(M, 'Laplacian', 1.0)
        pressure = fvm.construct(M, 'bodyForce')

        momentumMatrix = -laplacianM - pressure

        #momentumMatrix.print(LHS=False,RHS=True)

        # Relax momentum sum matrix
        #momentumMatrix.relax(relaxation=0.95)

        # Solve momentum sum matrix
        momentumMatrix.solve()

        #print(M._cellValues)

        # Assemble the deflection matrix
        laplacianW = fvm.construct(w, 'Laplacian', bendingStiffness)
        sourceW = fvm.construct(M, 'Su')

        wMatrix = laplacianW + sourceW

        #wMatrix.print(LHS=False, RHS=True)

        # Relax deflection matrix
        #wMatrix.relax(relaxation=0.95)

        # Solve deflection matrix
        wMatrix.solve()

        iCorr += 1
        if iCorr > 12:
            break;

    # Write results
    w.write(solControl.time())

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')