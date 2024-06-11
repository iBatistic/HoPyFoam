"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__ = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'
__all__ = ['Laplacian']

from src.foam.foamFileParser import *
from petsc4py import PETSc

ADD = PETSc.InsertMode.ADD_VALUES

class Laplacian():

    @classmethod
    def construct(self, psi, gamma):

        print('Assembling system of equations for Laplacian operator\n')

        try:
            if (psi._dimensions == 1):
                return self.scalarLaplacian(psi, gamma)
            elif (psi._dimensions == 3):
                return self.vectorLaplacian(psi, gamma)
            else:
                raise ValueError(f'Psi dimensions are set to {psi._dimensions}')
        except ValueError as e:
                print(f'Supported psi dimensions for Laplacian operator are scalar and vector')
                sys.exit(1)

    @classmethod
    def scalarLaplacian(self, psi, gamma):

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
                    A.setValues(cellP, cellIndex, value, ADD)
                    A.setValues(cellN, cellIndex, - value, ADD)

        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # Loop over paches (treat boundary faces).
        # Depending on boundary type call corresponding patch function and add source and diag terms
        for patch in psi._boundaryConditionsDict:

            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("LaplacianBoundaryConditions." + patchType)
            patchContribution(self, psi, mesh, source, A, patch, gamma)

        # Finish matrix assembly
        A.assemble()
        source.assemble()

        return source, A

    @classmethod
    def vectorLaplacian(self, psi, gamma):

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
                cx = psi.LRE().coeffs()[faceI][i]

                # Loop over Gauss point interpolation stencil and add
                # stencil cells contribution to matrix
                for j, cellIndex in enumerate(faceStencil):

                    # Owner and neighbour of current face
                    cellP = owner[faceI]
                    cellN = neighbour[faceI]
                    scalarVal = gammaMagSf * gpW * (cx[j] @ nf)
                    tenValue = scalarVal * np.array([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=PETSc.ScalarType)

                    # Store tensorial Laplace coefficients
                    A.setValuesBlocked(cellP, cellIndex, tenValue.flatten(), addv=ADD)
                    A.setValuesBlocked(cellN, cellIndex, -tenValue.flatten(), addv=ADD)

        # Treat boundaries
        source = A.createVecLeft()
        source.set(0.0)
        source.setUp()

        # Loop over paches (treat boundary faces).
        # Depending on boundary type call corresponding patch function and add source and diag terms
        for patch in psi._boundaryConditionsDict:
            # Get patch type
            patchType = psi._boundaryConditionsDict[patch]['type']

            # Call corresponding patch function
            patchContribution = eval("LaplacianBoundaryConditions." + patchType)
            patchContribution(self, psi, mesh, source, A, patch, gamma)

        # Assemble the matrix and the RHS vector
        A.assemble()
        source.assemble()

        # A.convert('dense')
        # test= A.getDenseArray()
        # np.set_printoptions(precision=2, suppress=True)
        # wr= [test[i:i + 1].tolist() for i in range(0, len(test), 1)]
        #
        # for item in wr:
        #     print(np.array(item))

        #A.view()
        #source.view()
        #array_A = A.convert('dense')
        #output_file = "Laplacianmatrix.txt"
        #np.savetxt(output_file, array_A, fmt='%0.2f')

        return source, A

class LaplacianBoundaryConditions(Laplacian):

    def empty(self, *args):
        pass

    def zeroGradient(self, *args):
        pass

    def solidTraction(self, psi, mesh, source, A, patch, gamma):
        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        owner = mesh._owner

        # Prescribed value at boundary
        prescribedValue = psi._boundaryConditionsDict[patch]['value']['uniform']

        # Loop over patch faces
        for faceI in range(startFace, startFace + nFaces):

            # Face area magnitude
            magSf = mesh.magSf[faceI]

            # Face owner
            cellP = owner[faceI]

            # Traction multiplied with face area to get force
            value = prescribedValue * magSf

            source.setValues(range(cellP * psi.dim, cellP * psi.dim + psi.dim), -value, ADD)

    def fixedValue(self, psi, mesh, source, A, patch, gamma):

        # Preliminaries
        startFace = mesh.boundary[patch]['startFace']
        nFaces = mesh.boundary[patch]['nFaces']

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights
        owner = mesh._owner

        # Prescribed value at boundary
        prescribedValue = psi._boundaryConditionsDict[patch]['value']['uniform']

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
                    scalarVal = gammaMagSf * gpW * (cx[j] @ nf)

                    if (psi.dim == 1):
                        # Scalar coeffs in case of scalar field
                        A.setValues(cellP, cellIndex, scalarVal, ADD)
                    else:
                        # Tensor coeffs in case of vector field
                        tenValue = scalarVal * np.array([[1, 0, 0],
                                                         [0, 1, 0],
                                                         [0, 0, 1]], dtype=PETSc.ScalarType)

                        # Store tensorial Laplace coefficients
                        A.setValuesBlocked(cellP, cellIndex, tenValue.flatten(), addv=ADD)

                    if (j == len(faceStencil) - 1):
                        # Boundary face centre is not included in face stencil list
                        # Its contribution is added last, after cell centres
                        value = gammaMagSf * gpW * (cx[j + 1] @ nf) * prescribedValue
                        source.setValues(range(cellP * psi.dim, cellP * psi.dim + psi.dim), -value, ADD)

    # Only implemented for scalar field, for example_2 boundary conditions
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
                        source.setValue(cellP, - value, ADD)
