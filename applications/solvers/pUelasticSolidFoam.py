"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Pressure based solver for incompressible solid with small strains and small
    rotations assumption (linear geometry). Total displacement is the primary
    unknown. The stress is calculated according to linear-elastic Hooke law.
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
from src.foam.field.volField.volScalarField import volScalarField
from src.finiteVolume.fvMatrices.fvm.fvm import fvm
from src.finiteVolume.fvMatrices.fvc.fvc import fvc
from src.finiteVolume.fvMatrices.fvc.operators.interpolate import interpolate


# Execution start time, used to measured elapsed clock time
exec_start_time = timeModule.perf_counter()

# Get command line arguments
args = arg_parser().parse_args()

# Read and construct mesh, construct interpolation stencil
mesh = fvMesh()

solControl = solutionControl()

# N is Taylor polynomial degree, 1 is second-order, 2 is third-order...
N = 1

# Initialise displacement vector field U, N is interpolation order
U = volVectorField("U", mesh, readVectorField("U"),  N)

# Initialise pressure scalar field p
p = volScalarField("p", mesh, readScalarField("p"), N)

# Initialise pressure correction scalar field pCorr
#pCorr = volScalarField("p", mesh, readScalarField("pCorr"), N)

# Read mechanichalProperties dict to get first and second Lame parameters
mu, lam = readMechanicalProperties()

# Move to SIMPLE or to be consistent with OF calculate here.
# MAybe add fvc.interpolate functions?
def predictFaceDisplacement(U, p, pressureGrad, D):
     '''
     Correct face displacement according to Rhie-chow interpolation.
     Return field is corrected displacement value for each Gauss point
     '''

   # Ustar = interpolate.vectorInterpolate(U)
   # rAU = interpolate.vectorInterpolate(1/D)
   # pressureGrad = interpolate.vectorInterpolate()
# interpolate.
#     pass


while (solControl.loop()):

    print(f'Time = {solControl.time()} \n')

    # Assemble the Laplacian, LaplacianTranspose and body force
    laplacian = fvm.construct(U, 'Laplacian', mu)
    laplacianTranspose = fvm.construct(U, 'LaplacianTranspose', mu)
    bodyForce = fvm.construct(U, 'bodyForce')

    # Explicit calculation of pressure grad, U and mu are required for
    # boundaries with specified pressure from constitutive relation
    pressureGrad = fvc.construct(p, 'grad', mu, U)

    momentumMatrix = laplacian + laplacianTranspose + bodyForce - pressureGrad

    # Solve momentum matrix
    momentumMatrix.solve()

    # Velocity interpolated to face Gauss points
    UStar = interpolate.interpolate(U)

    # Inverted momentum matrix diagonal coefficients multiplied with cell volume
    #volOverDiag = momentumMatrix.volOverD()
    #pGamma = interpolate.scalarInterpolate(volOverDiag)

    #divU = fvc.construct(UStar, 'div')
    #laplacianPressureCorr = fvm.construct(pCorr, 'pressureCorrLaplacian', pGamma)

    #pressureCorrMatrix = divU - laplacianPressureCorr

    # Solve pressure correction matrix
    #pressureCorrMatrix.solve()

    # Correct pressure and displacement fields
    #p = p + pCorr
    #gradPCorr = fvc.construct(pcorrGrad--)
    #U = UStar - volOverDiag@gradPCorr


    # momentumMatrix._A.convert('dense')
    # test= momentumMatrix._A.getDenseArray()
    # np.set_printoptions(precision=2, suppress=True)
    # wr= [test[i:i + 1].tolist() for i in range(0, len(test), 1)]
    #
    # for item in wr:
    #     print(np.array(item))
    # # Displacement correction using Rhie-Chow interpolation
    # D = momentumMatrix.A
    # UHat = predictFaceDisplacement(U, p, pressureGrad, D)
    #
    # # Pressure equation
    # divU = fvm.construct(UHat, 'div')
    # pressureLaplacian = fvm.construct(pCorr, 'Laplacian', D)
    #
    # Write results
    U.write(solControl.time())

    print(f'Execution time = '
          f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')