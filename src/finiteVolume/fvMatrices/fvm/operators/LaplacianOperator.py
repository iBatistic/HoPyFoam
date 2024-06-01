"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description

"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['LaplacianOperator']

import os
import sys
from src.foam.foamFileParser import *
from petsc4py import PETSc


class LaplacianOperator():

    @classmethod
    def Laplacian(self, psi, gamma):

        print('Assembling system of equations for Laplacian operator\n')

        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes((nCells, nCells))
        A.setType(PETSc.Mat.Type.AIJ)

        # Allow matrix options set from options file
        A.setFromOptions()

        # Estimate the number of nonzeros to be expected on each row
        # This is not working as it should, input is how wide the diagonal, depends on numbering
        #A.setPreallocationNNZ(psi.Nn)

        owner = mesh._owner
        neighbour = mesh._neighbour

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over internal faces
        for faceI in range(nInternalFaces):

            # Diffusivity coefficient multiplied with face magnitude
            gammaMagSf = mesh.magSf[faceI] * gamma

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                # Gauss point gp weight
                gpW = faceGaussPointsAndWeights[0][i]

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cx = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    # Owner and neighbour of current face
                    cellP = owner[faceI]
                    cellN = neighbour[faceI]
                    value = gammaMagSf * gpW * (cx[j] @ nf)

                    # Store Laplace coefficients
                    A.setValues(cellP, cellIndex, value, addv=PETSc.InsertMode.ADD_VALUES)
                    A.setValues(cellN, cellIndex, - value, addv=PETSc.InsertMode.ADD_VALUES)

        source = A.createVecLeft()
        source.set(0.0)

        # Loop over paches (treat boundary faces).
        # Depending on boundary type call corresponding patch function and add source and diag terms
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("LaplacianOperatorBoundaryConditions." + patchType)
            patchContribution(self, psi, mesh, source, A, patch, gamma)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        #A.view(PETSc.Viewer("APETSc.mat", 'w'))
        return source, A

class LaplacianOperatorBoundaryConditions(LaplacianOperator):

    def empty(self, *args):
        pass

    def zeroGradient(self, *args):
        pass

    def fixedValue(self, psi, mesh, source, A, patch, gamma):

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        owner = mesh._owner

        # Prescribed value at boundary
        prescribedValue = convert_to_float(psi._boundaryConditionsDict[patch]['value']['uniform'])

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Diffusivity coefficient multiplied with face magnitude
            gammaMagSf = mesh.magSf[faceI] * gamma

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                # Gauss point gp weight
                gpW = faceGaussPointsAndWeights[0][i]

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cx = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    cellP = owner[faceI]
                    value = gammaMagSf * gpW * (cx[j] @ nf)
                    A.setValues(cellP, cellIndex, value, addv=PETSc.InsertMode.ADD_VALUES)

                    if (j == len(faceStencil) - 1):
                        # Boundary face centre is not included in face stencil list
                        # Its contribution is added last, after cell centres
                        value = gammaMagSf * gpW * (cx[j+1] @ nf) * prescribedValue
                        source.setValue(cellP, - value, addv=PETSc.InsertMode.ADD_VALUES)

    def analyticalFixedValue(self, psi, mesh, source, A, patch, gamma):
        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        owner = mesh._owner

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Diffusivity coefficient multiplied with face magnitude
            gammaMagSf = mesh.magSf[faceI] * gamma

            # Face normal
            nf = mesh.nf[faceI]

            # List of face Gauss points [1] and weights [0]
            faceGaussPointsAndWeights = GaussPointsAndWeights[faceI]

            # Current face points interpolation stencil
            faceStencil = psi._facesInterpolationMolecule[faceI]

            # Loop over Gauss points
            for i, gp in enumerate(faceGaussPointsAndWeights[1]):

                # Gauss point gp weight
                gpW = faceGaussPointsAndWeights[0][i]

                # Gauss point interpolation coefficient vector for each neighbouring cell
                cx = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    cellP = owner[faceI]
                    value = gammaMagSf * gpW * (cx[j] @ nf)
                    A.setValues(cellP, cellIndex, value, addv=PETSc.InsertMode.ADD_VALUES)

                    if (j == len(faceStencil) - 1):

                        # Hard-coded value at boundary
                        x = gp[0]
                        y = gp[1]
                        presribedValue = np.sin(5*y) * np.exp(5*x)

                        value = gammaMagSf * gpW * (cx[j+1] @ nf) * presribedValue
                        # Boundary face centre is not included in face stencil list
                        # Its contribution is added last, after cell centres
                        source.setValue(cellP, - value, addv=PETSc.InsertMode.ADD_VALUES)

