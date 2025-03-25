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

# Read mechanichalProperties dict to get Young modulus and Poisson's ratio
E, nu = readMechanicalProperties(convertToLameCoeffs=False)

# Read solidProperties dict to get plate thickness, nCorrectors and altTol.
h, nCorrectors, alternativeTolerance = readSolidProperties()

bendingStiffness = E*pow(h,3)/(12*(1-pow(nu,2)))

# Initialise deflection scalar field w, N is interpolation order
w = volScalarField("w", mesh, readScalarField("w"),  N=3)

# Initialise moment sum  scalar field M, N is interpolation order
M = volScalarField("M", mesh, readScalarField("M"),  N=3)

while (solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    iCorr = 0
    print("\tCorr, relRes M, relRes w, solRes M, solRes w")
    while True:

        # Assemble the momentum sum matrix
        laplacianM = fvm.construct(M, 'Laplacian', 1.0)
        pressure = fvm.construct(M, 'bodyForce')

        momentumMatrix = -laplacianM - pressure

        # Relax momentum sum matrix
        momentumMatrix.relax(relaxation=1)

        # Solve momentum sum matrix
        solverPerfM = momentumMatrix.solve()

        # Assemble the deflection matrix
        laplacianW = fvm.construct(w, 'Laplacian', bendingStiffness)
        sourceW = fvm.construct(M, 'Su')

        wMatrix = laplacianW + sourceW

        # Relax deflection matrix
        wMatrix.relax(relaxation=1)

        # Solve deflection matrix
        solverPerfW = wMatrix.solve()

        momentumResidual =\
            np.max\
            (
                np.abs(np.array(M._cellValues)-np.array(M.prevIter))
                / max(np.max(np.abs(M._cellValues)), 1e-10)
            )

        wResidual =\
            np.max\
            (
                np.abs(np.array(w._cellValues) - np.array(w.prevIter))
                / max(np.max(np.abs(w._cellValues)), 1e-10)
            )

        print(f'\tIter {iCorr}, {momentumResidual:.5g}, {wResidual:.5g}, '
              f'{solverPerfM["residualNorm"]:.5g}, '
              f'{solverPerfW["residualNorm"]:.5g}')

        iCorr += 1
        if (iCorr > nCorrectors or
                (momentumResidual < alternativeTolerance
                 and wResidual < alternativeTolerance)):
            break;


    # Write results
    w.write(solControl.time())

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')