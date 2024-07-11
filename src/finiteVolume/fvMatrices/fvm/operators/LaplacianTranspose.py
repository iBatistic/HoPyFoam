"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['LaplacianTranspose']

from src.foam.foamFileParser import *
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class LaplacianTranspose():

    @classmethod
    def construct(self, psi, gamma):

        mesh = psi._mesh
        nCells = mesh.nCells
        nInternalFaces = mesh.nInternalFaces

        # Read PETSc options file
        OptDB = PETSc.Options()

        nblocks = nCells
        blockSize = psi._dimensions

        # total number of rows or columns
        matSize = nblocks * blockSize

        A = PETSc.Mat()
        A.create(comm=PETSc.COMM_WORLD)
        A.setSizes([matSize, matSize])
        A.setType(PETSc.Mat.Type.BAIJ)
        A.setBlockSize(blockSize)

        A.setFromOptions()
        A.setUp()

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
                cx = psi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    # Owner and neighbour of current face
                    cellP = owner[faceI]
                    cellN = neighbour[faceI]

                    nfx = nf[0]
                    nfy = nf[1]
                    nfz = nf[2]
                    cxx = cx[j][0]
                    cxy = cx[j][1]
                    cxz = cx[j][2]
                    tenValue = np.array([[cxx*nfx, cxx*nfy, cxx*nfz],
                                        [cxy*nfx, cxy*nfy, cxy*nfz],
                                        [cxz*nfx, cxz*nfy, cxz*nfz]], dtype=PETSc.ScalarType)

                    tenValue *= gammaMagSf * gpW

                    # Store tensorial Laplace coefficients
                    A.setValuesBlocked(cellP, cellIndex, tenValue.flatten(), addv=ADD)
                    A.setValuesBlocked(cellN, cellIndex, -tenValue.flatten(), addv=ADD)

        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        for patch in psi._boundaryConditionsDict:
            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("LaplacianTransposeBoundaryConditions." + patchType)
            patchContribution(self, psi, mesh, source, A, patch, gamma)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        # A.view(PETSc.Viewer("APETSc.mat", 'w'))
        return source, A

class LaplacianTransposeBoundaryConditions(LaplacianTranspose):

    def empty(self, *args):
        pass

    def zeroGradient(self, *args):
        pass

    def solidTraction(self, *args):
        pass

    def fixedValue(self, psi, mesh, source, A, patch, gamma):

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        owner = mesh._owner

        # Prescribed value at boundary
        prescribedValue = np.array(psi._boundaryConditionsDict[patch]['value']['uniform'])

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
                cx = psi.LRE().gradCoeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    # Owner and neighbour of current face
                    cellP = owner[faceI]

                    nfx = nf[0]
                    nfy = nf[1]
                    nfz = nf[2]
                    cxx = cx[j][0]
                    cxy = cx[j][1]
                    cxz = cx[j][2]
                    tenValue = np.array([[cxx*nfx, cxx*nfy, cxx*nfz],
                                        [cxy*nfx, cxy*nfy, cxy*nfz],
                                        [cxz*nfx, cxz*nfy, cxz*nfz]], dtype=PETSc.ScalarType)

                    tenValue *= gammaMagSf * gpW

                    # Store tensorial Laplace coefficients
                    A.setValuesBlocked(cellP, cellIndex, tenValue.flatten(), addv=ADD)

                    if (j == len(faceStencil) - 1):
                        # Boundary face centre is not included in face stencil list
                        # Its contribution is added last, after cell centres\
                        nfx = nf[0]
                        nfy = nf[1]
                        nfz = nf[2]
                        cxx = cx[j + 1][0]
                        cxy = cx[j + 1][1]
                        cxz = cx[j + 1][2]
                        tenValue = np.array([[cxx * nfx, cxx * nfy, cxx * nfz],
                                             [cxy * nfx, cxy * nfy, cxy * nfz],
                                             [cxz * nfx, cxz * nfy, cxz * nfz]], dtype=PETSc.ScalarType)

                        tenValue *= gammaMagSf * gpW

                        value = np.dot(tenValue, prescribedValue)
                        source.setValues(range(cellP * psi.dim, cellP * psi.dim + psi.dim), -value, ADD)