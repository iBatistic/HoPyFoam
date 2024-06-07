from petsc4py import PETSc
import numpy as np


def main_():
    # Create a block matrix (2x2 blocks of 3x3 matrices)
    # nblocks = 2
    # block_size = 3
    # m = nblocks * block_size  # total number of rows
    # n = nblocks * block_size  # total number of columns
    #
    # A = PETSc.Mat().create()
    # A.setSizes([m, n])
    # A.setType(PETSc.Mat.Type.MPIBAIJ)
    # A.setBlockSize(block_size)
    # A.setUp()
    #
    # # Define blocks
    # block1 = np.array([[1, 2, 3],
    #                    [4, 5, 6],
    #                    [7, 8, 9]], dtype=PETSc.ScalarType)
    # block2 = np.array([[9, 8, 7],
    #                    [6, 5, 4],
    #                    [3, 2, 1]], dtype=PETSc.ScalarType)
    #
    # # Set values in the matrix
    # # The block values need to be provided as a flat array
    # blocks = np.array([block1.flatten(), block2.flatten()])
    #
    # # Loop over the block rows and columns to set the values
    # for i in range(nblocks-1):
    #     for j in range(nblocks-1):
    #         A.setValuesBlocked(i, j, blocks[i * nblocks + j])
    #
    # # Assemble the matrix
    #A.assemblyBegin()
    #A.assemblyEnd()
    #Initialize PETSc
    #PETSc.Initialize()

    #Create a 2x2 block matrix, where each block is a 2x2 matrix

    nblocks = 2
    block_size = 3

    m = nblocks * block_size  # total number of rows
    n = nblocks * block_size  # total number of columns
    #
    # A = PETSc.Mat().create()
    # A.setSizes([m, n])

    # Total matrix size
    n = nblocks * block_size

    # Create a block matrix of size 4x4 (2x2 blocks of 2x2 matrices)
    A = PETSc.Mat()
    A.create(comm=PETSc.COMM_WORLD)
    A.setSizes([m, n])
    A.setType(PETSc.Mat.Type.BAIJ)
    A.setBlockSize(block_size)

    A.setFromOptions()
    A.setUp()

    # Insert values into the block matrix
    # Blocks are indexed by (block_row, block_col), values are 2x2 arrays
    #block_values = [[1, 2], [3, 4]]
    #for i in range(n_blocks):
    #    for j in range(n_blocks):
    #        A.setValuesBlocked(i, j, block_values, addv=PETSc.InsertMode.ADD_VALUES)
    #A.setValuesBlocked([0,1], [0,1], [1,2,3,4])#, addv=PETSc.InsertMode.ADD_VALUES)

    block = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=PETSc.ScalarType)

    A.setValuesBlocked([0], [0], block.flatten(), addv=PETSc.InsertMode.ADD_VALUES)
    A.setValuesBlocked([0], [1], block.flatten(), addv=PETSc.InsertMode.ADD_VALUES)
    A.setValuesBlocked([1], [0], block.flatten(), addv=PETSc.InsertMode.ADD_VALUES)
    A.setValuesBlocked([1], [1], block.flatten(), addv=PETSc.InsertMode.ADD_VALUES)

    # Assemble the matrix
    A.assemble()
    A.view()
    # Create vectors for the solution (x) and RHS (b)
    x = PETSc.Vec().create()
    total_size = nblocks*block_size
    x.setSizes(total_size)
    x.setUp()

    b = PETSc.Vec().create()
    b.setSizes(total_size)
    b.setUp()

    # Set values for the RHS vector b
    b.setValues(range(total_size), np.zeros(total_size, dtype=PETSc.ScalarType))

    # Assemble the RHS vector
    b.assemblyBegin()
    b.assemblyEnd()

    # Create the linear solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    # Set solver type and options (optional)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    # Solve the system Ax = b
    ksp.solve(b, x)

    # Get the solution
    solution = x.getArray()

    # Print the solution
    print("Solution:", solution)

    # Zbrajanje matrica
    # M = B.__add__(A)

    # Print the matrix to stdout


    # Finalize PETSc
    #PETSc.Finalize()
\

def main():

    # Define block size and matrix size
    block_size = 3
    num_blocks = 2
    total_size = block_size * num_blocks

    # Create a block matrix
    A = PETSc.Mat().create()
    A.setSizes([total_size, total_size])
    A.setType(PETSc.Mat.Type.BAIJ)
    A.setBlockSize(block_size)
    A.setUp()

    # Define blocks
    B00 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=PETSc.ScalarType)
    B01 = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]], dtype=PETSc.ScalarType)
    B10 = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]], dtype=PETSc.ScalarType)
    B11 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=PETSc.ScalarType)

    # Set values in the matrix using block indices and flattened blocks
    A.setValuesBlocked([0], [0], B00.flatten())
    A.setValuesBlocked([0], [1], B01.flatten())
    A.setValuesBlocked([1], [0], B10.flatten())
    A.setValuesBlocked([1], [1], B11.flatten())

    # Assemble the matrix
    A.assemblyBegin()
    A.assemblyEnd()

    # Create vectors for the solution (x) and RHS (b)
    x = PETSc.Vec().create()
    x.setSizes(total_size)
    x.setUp()

    b = PETSc.Vec().create()
    b.setSizes(total_size)
    b.setUp()

    # Set values for the RHS vector b
    b.setValues(range(total_size), np.array([1, 2, 3, 4, 5, 6], dtype=PETSc.ScalarType))

    b.setValues([0,1,2], np.array([0,2,3]),  addv=PETSc.InsertMode.ADD_VALUES)

    # Assemble the RHS vector
    b.assemblyBegin()
    b.assemblyEnd()

    # Create the linear solver
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)

    # Set solver type and options (optional)
    ksp.setType(PETSc.KSP.Type.CG)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)

    A.view()
    b.view()

    # Solve the system Ax = b
    ksp.solve(b, x)

    # Get the solution
    solution = x.getArray()

    # Print the solution
    print("Solution:", solution)


if __name__ == "__main__":
    main()