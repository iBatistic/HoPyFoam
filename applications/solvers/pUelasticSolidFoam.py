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

# Read mechanichalProperties dict to get Lame parameters
mu, lam = readMechanicalProperties()

# N is Taylor polynomial degree, 1 is second-order, 2 is third-order...
N = 1

# Initialise displacement cell-centred vector field U
U = volVectorField("U", mesh, readVectorField("U"),  N)

# Displacement surface vector field at face Gauss points
Uf = surfaceVectorField("Uf", U)

# Initialise pressure cell-centred scalar field p
p = volScalarField("p", mesh, readScalarField("p"), N, True)

# Surface scalar pressure field at face Gauss points
pf = surfaceScalarField("pf", p, U, mu)

# Initialise pressure correction cell-centred scalar field
pCorr = volScalarField("pCorr", mesh, readScalarField("pCorr"), N)

# Surface scalar pressure correction field at face Gauss points
pCorrF = surfaceScalarField("pCorrF", pCorr)

# Face pressure gradient, calculated using interpolation coefficients
pGradF = surfaceVectorGradField("pGradF", p, pf)

# Face pressure gradient, interpolated gradient at cell centre to face points
pBarGradF = surfaceVectorBarGradField("pBarGradF", p, pf)

while (solControl.loop()):

    print(f'\nTime = {solControl.time()} \n')

    # Assemble the Laplacian, LaplacianTranspose and body force
    laplacian = fvm.construct(U, 'Laplacian', mu)
    laplacianTranspose = fvm.construct(U, 'LaplacianTranspose', mu)
    bodyForce = fvm.construct(U, 'bodyForce')

    # Update pressure value at face Gauss points
    pf.evaluate()

    # Explicit calculation of pressure grad using Gauss theorem
    pressureGrad = fvc.construct(pf, 'grad')

    #pCellGrad = pressureGrad._source.getArray()
    #pCellGrad = [pCellGrad[i:i + 3].tolist() for i in range(0, len(pCellGrad), 3)]
    # print("Cell grad for 9", pCellGrad[9])
    # print("Cell grad for 19", pCellGrad[19])
    # print("Cell grad for 29", pCellGrad[29])
    # print("Cell grad for 39", pCellGrad[39])

    momentumMatrix = laplacian + laplacianTranspose + bodyForce - pressureGrad

    # Relax momentum matrix
    momentumMatrix.relax(relaxation=0.95)

    # Solve momentum matrix
    momentumMatrix.solve()

    # Update displacement at face Gauss points
    Uf.evaluate()

    # Inverse of momentum matrix diagonal (tensor coefficient) multiplied with
    # cell volume
    volOverDiag = momentumMatrix.volOverD()

    # Inverse of momentum matrix diagonal at face Gauss points
    volOverDiagF = momentumMatrix.volOverDf()

    # Update pressure field because of pressureTraction boundary
    pf.evaluate()

    # Face Gauss points pressure gradient using interpolation coefficients
    pGradF.evaluate()
    pBarGradF.evaluate()

    # Rhie-Chow correction for interpolated velocity
    UHat = np.copy(Uf._faceValues)
    for faceI in range(mesh.nFaces):
         for gpI in range(U.Ng):
                UHat[faceI][gpI] += volOverDiagF[faceI][gpI] @ (pBarGradF[faceI][gpI] - pGradF[faceI][gpI])

    divU = fvc.construct(UHat, 'div', cellPsi=U)
    laplacianPressureCorr = fvm.construct(pCorr, 'pCorrLaplacian', volOverDiagF)

    pressureCorrMatrix = laplacianPressureCorr + divU

    # Solve pressure correction matrix
    pressureCorrMatrix.solve()

    # Correct pressure field
    p.correct(pCorr, relaxation=0.1)

    for cell in range(len(pCorr._cellValues)):
        pCorr._cellValues[cell][0] *= 0.1

    # Correct displacement field
    pCorrF.evaluate()
    pCorrCellGrad = fvc.construct(pCorrF, 'grad')._source.getArray()
    pCorrCellGrad = [pCorrCellGrad[i:i + 3].tolist() for i in range(0, len(pCorrCellGrad), 3)]

    for cellI, pC in enumerate(pCorrCellGrad):
        U._cellValues[cellI] -= np.dot(volOverDiag[cellI], pC)

    # Write results
    U.write(solControl.time())
    p.write(solControl.time(), pf)
    pCorr.write(solControl.time())
    pGradF.write(solControl.time())
    pBarGradF.write(solControl.time())

    # Check convergence and stop iterating if convergence is meet
    #break

print(f'\nExecution time = '
      f'{timeModule.perf_counter() - exec_start_time:.2f} s \n')

print('End\n')