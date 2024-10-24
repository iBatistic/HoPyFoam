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
from src.foam.field.volField import *
from src.foam.field.surfaceField import *
from src.finiteVolume.fvMatrices.fvm.fvm import fvm
from src.finiteVolume.fvMatrices.fvc.fvc import fvc
from src.finiteVolume.fvMatrices.fvc.operators import *

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

volOverD = volVectorField("volOverD", mesh, readVectorField("volOverD"),  N)

# Initialise pressure scalar field p
p = volScalarField("p", mesh, readScalarField("p"), N)

# Initialise pressure scalar field p
pBar = volScalarField("pBar", mesh, readScalarField("pBar"), N)

# Initialise pressure correction scalar field pCorr
pCorr = volScalarField("pCorr", mesh, readScalarField("pCorr"), N)

# Read mechanichalProperties dict to get first and second Lame parameters
mu, lam = readMechanicalProperties()

while (solControl.loop()):

    print(f'\nTime = {solControl.time()} \n')

    # Assemble the Laplacian, LaplacianTranspose and body force
    laplacian = fvm.construct(U, 'Laplacian', mu)
    laplacianTranspose = fvm.construct(U, 'LaplacianTranspose', mu)
    bodyForce = fvm.construct(U, 'bodyForce')

    # Explicit calculation of pressure grad, U and mu are required for
    # boundaries with specified pressure from constitutive relation
    pressureGrad = fvc.construct(p, 'grad', mu, U)

    momentumMatrix = laplacian + laplacianTranspose + bodyForce - pressureGrad

    # Relax momentum matrix
    momentumMatrix.relax(relaxation=0.95)

    # Solve momentum matrix
    momentumMatrix.solve()
    #sys.exit(1)
    # Displacement interpolated to face Gauss points
    UStar = interpolate.cellToFace(U)

    # Inverse of momentum matrix diagonal (tensor coefficient) multiplied with
    # cell volume
    volOverDiag = momentumMatrix.volOverD()

    # Interpolate volOverDiag to face Gauss points
    pGamma = interpolate.diagToFace(volOverDiag, volOverD)

    # Face Gauss points pressure gradient using interpolation coefficients
    # mu and U are used for pressureTraction boundaries to get pressure from
    # constitutive relation
    pressureGrad = interpolateGrad.cellToFace(p, mu, U)
    barPressureGrad = interpolateCellGrad.cellToFace(p, mu, U, pBar)

    pgf = surfaceVectorField("pGrad", pressureGrad, p)
    pbgf = surfaceVectorField("pBarGrad", barPressureGrad, pBar)
    pgf.write(solControl.time())
    pbgf.write(solControl.time())

    # Rhie-Chow correction for interpolated velocity
    UHat = UStar

    for faceI in range(mesh.nFaces):
        for gpI in range(U.Ng):
            UHat[faceI][gpI] += pGamma[faceI][gpI] @ (barPressureGrad[faceI][gpI] - pressureGrad[faceI][gpI] )

    divU = fvc.construct(UHat, 'div', cellPsi=U)
    laplacianPressureCorr = fvm.construct(pCorr, 'pCorrLaplacian', pGamma)

    pressureCorrMatrix = laplacianPressureCorr + divU

    # Solve pressure correction matrix
    pressureCorrMatrix.solve()

    # Correct pressure field
    p.addCorrection(pCorr, relaxation=0.09)

    # Correct displacement field
    pCorrCellGrad = fvc.construct(pCorr, 'grad')._source.getArray()
    pCorrCellGrad = [pCorrCellGrad[i:i + 3].tolist() for i in range(0, len(pCorrCellGrad), 3)]

    for cellI, pC in enumerate(pCorrCellGrad):
       U._cellValues[cellI] -= volOverDiag[cellI] @ pC

    # Write results
    U.write(solControl.time())
    #p.evaluateBoundary()
    p.write(solControl.time())
    pCorr.write(solControl.time())

    # Check convergence and stop iterating if convergence is meet
    #break

print(f'\nExecution time = '
      f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')