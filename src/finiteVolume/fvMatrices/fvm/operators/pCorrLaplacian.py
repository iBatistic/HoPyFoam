"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['pCorrLaplacian']

from src.foam.foamFileParser import *
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class pCorrLaplacian():

    @classmethod
    def construct(self, psi, gamma):

        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes((nCells, nCells))
        A.setType(PETSc.Mat.Type.AIJ)
        A.setUp()

        # Allow matrix options set from options file
        A.setFromOptions()

        # Estimate the number of nonzeros to be expected on each row
        # This is not working as it should, input is how wide the diagonal, depends on numbering
        #A.setPreallocationNNZ(psi.Nn)
        #diag = 0

        owner = mesh._owner
        neighbour = mesh._neighbour

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over internal faces
        for faceI in range(nInternalFaces):

            # Diffusivity coefficient multiplied with face magnitude
            magSf = mesh.magSf[faceI]

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
                cx = psi.LRE().internalGradCoeffs[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    # Owner and neighbour of current face
                    cellP = owner[faceI]
                    cellN = neighbour[faceI]
                    value = gpW * magSf * np.dot(np.dot(gamma[faceI][i], cx[j]), nf)
                                        # (gamma[faceI][i] @ cx[j]) @ nf
                    # Store Laplace coefficients
                    A.setValues(cellP, cellIndex, value, ADD)
                    A.setValues(cellN, cellIndex, - value, ADD)

                    # if cellIndex == 15:
                    #     if cellP == 15:
                    #         diag += value
                    #     if cellN == 15:
                    #         diag += -value

        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # Loop over paches (treat boundary faces).
        # Depending on boundary type call corresponding patch function and add source and diag terms
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("pCorrLaplacianBoundaryConditions." + patchType)
            patchContribution(self, psi, mesh, source, A, patch, gamma)#, diag)

        #val = 200
        #diag = A.getValue(15, 15)  # )zeroRows([15], diag=1.0)
        #print(diag)
        #A.setValues(15, 15, diag, ADD )
        #source.setValue(15, diag * val)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A


class pCorrLaplacianBoundaryConditions(pCorrLaplacian):

    def empty(self, *args):
        pass

    def zeroGradient(self, *args):
        """Zero normal gradient boundary condition"""
        pass

    def fixedValue(self, psi, mesh, source, A, patch, gamma):
        """
        Fixed value - zero correction for pressure field, prescribed value is 0
        """

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        owner = mesh._owner

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Diffusivity coefficient multiplied with face magnitude
            magSf = mesh.magSf[faceI]

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
                cx = psi.LRE().boundaryGradCoeffsGhost[faceI - mesh.nInternalFaces][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    cellP = owner[faceI]
                    value = gpW * magSf * np.dot(np.dot(gamma[faceI][i], cx[j]), nf)

                    A.setValues(cellP, cellIndex, value, ADD)

                    # if cellP == 15:
                    #     if cellIndex == 15:
                    #         diag += value