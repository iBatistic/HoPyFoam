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
__all__ = ['div']


import os
from src.foam.foamFileParser import *
from petsc4py import PETSc
from src.finiteVolume.fvMatrices.fvc.operators.interpolate import interpolate

ADD = PETSc.InsertMode.ADD_VALUES

class div():

    @classmethod
    def construct(self, GaussPointsValues, gamma, cellPsi):

        mesh = cellPsi._mesh
        nCells = mesh.nCells

        # Read PETSc options file
        OptDB = PETSc.Options()

        # Create empty left left-hand side matrix
        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes((nCells, nCells))
        A.setType(PETSc.Mat.Type.AIJ)
        A.setUp()

        # Create solution vector
        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # First loop over internal faces
        for faceI in range(mesh.nInternalFaces):

            # Face magnitude
            magSf = mesh.magSf[faceI]

            # Face normal
            nf = mesh.nf[faceI]

            faceGaussPointsAndWeights = cellPsi._facesGaussPointsAndWeights[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                # Gauss point gp weight
                gpW = faceGaussPointsAndWeights[0][i]

                # Owner and neighbour of current face
                cellP = mesh._owner[faceI]
                cellN = mesh._neighbour[faceI]
                value = gpW * magSf * (GaussPointsValues[faceI][i] @ nf)

                source.setValues(cellP, value, ADD)
                source.setValues(cellN, -value, ADD)

        # Loop over boundary faces
        for patch in cellPsi._boundaryConditionsDict:
            startFace = mesh.boundary[patch]['startFace']
            nFaces = mesh.boundary[patch]['nFaces']

            # Loop over patch faces
            for faceI in range(startFace, startFace + nFaces):

                # Face magnitude
                magSf = mesh.magSf[faceI]

                # Face normal
                nf = mesh.nf[faceI]

                faceGaussPointsAndWeights = cellPsi._facesGaussPointsAndWeights[faceI]

                # Loop over Gauss points
                for i, gp in enumerate(faceGaussPointsAndWeights[1]):
                    # Gauss point gp weight
                    gpW = faceGaussPointsAndWeights[0][i]

                    # Owner and neighbour of current face
                    cellP = mesh._owner[faceI]
                    value = gpW * magSf * (GaussPointsValues[faceI][i] @ nf)

                    source.setValues(cellP, value, ADD)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        #source.view()
        return source, A