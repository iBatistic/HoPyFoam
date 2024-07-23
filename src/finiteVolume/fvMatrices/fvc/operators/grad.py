"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Explicit calculation of gradient at cell centres
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['grad']


import os
from src.foam.foamFileParser import *
from petsc4py import PETSc
from src.finiteVolume.fvMatrices.fvc.operators.interpolate import interpolate

ADD = PETSc.InsertMode.ADD_VALUES

class grad():

    @classmethod
    def construct(self, psi, gamma, secondPsi):
        try:
            if (psi.dim == 1):
                return self.scalarGrad(psi, gamma, secondPsi)
            elif (psi.dim == 3):
                return self.vectorGrad(psi, gamma, secondPsi)
            else:
                raise ValueError(f'Psi dimensions are set to {psi._dimensions}')
        except ValueError as e:
                print(f'Supported psi dimensions for gradient operator are '
                      f'scalar and vector')
                sys.exit(1)

    @classmethod
    def vectorGrad(self, psi, gamma, secondPsi):
        pass

    @classmethod
    def scalarGrad(self, psi, gamma, secondPsi):
        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

        nblocks = nCells
        blockSize = psi.dim * 3
        matSize = nblocks * blockSize

        # Create empty left left-hand side matrix
        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes([matSize, matSize])
        A.setType(PETSc.Mat.Type.BAIJ)
        A.setBlockSize(blockSize)
        A.setFromOptions()
        A.setUp()

        # Create solution vector
        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # 1 stage: Interpolate pressure from cell centres to face Gauss points
        GaussPointsValues = interpolate.interpolate(psi, gamma, secondPsi)

        # 2 stage: cell-centre gradient using Gauss theorem
        # First loop over internal faces
        for faceI in range(mesh.nInternalFaces):

            # Face magnitude
            magSf = mesh.magSf[faceI]

            # Face normal
            nf = mesh.nf[faceI]

            faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                # Gauss point gp weight
                gpW = faceGaussPointsAndWeights[0][i]

                # Owner and neighbour of current face
                cellP = mesh._owner[faceI]
                cellN = mesh._neighbour[faceI]
                value = nf * magSf * GaussPointsValues[faceI][i] * gpW

                source.setValues(range(cellP * 3, cellP * 3 + 3), value, ADD)
                source.setValues(range(cellN * 3, cellN * 3 + 3), -value, ADD)

        # Loop over boundary faces
        for patch in psi._boundaryConditionsDict:
            # Preliminaries
            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            owner = mesh._owner

            # Loop over patch faces
            for faceI in range(startFace, startFace + nFaces):

                # Face area magnitude and unit normal vector
                magSf = mesh.magSf[faceI]
                nf = mesh.nf[faceI]

                # Face owner
                cellP = owner[faceI]

                faceGaussPointsAndWeights = psi._facesGaussPointsAndWeights[faceI]

                # Loop over Gauss points
                for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                    gpW = faceGaussPointsAndWeights[0][i]

                    value = nf * magSf * GaussPointsValues[faceI][i] * gpW
                    source.setValues(range(cellP * 3, cellP * 3 + 3), value, ADD)


        # Finish matrix assembly
        A.assemble()
        source.assemble()

        #source.view()
        return source, A