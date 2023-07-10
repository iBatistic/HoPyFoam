import math

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

'''
--------------------------------------------------------------------------------
                               INPUT DATA
--------------------------------------------------------------------------------
'''
# Overall rod length, cross-section area and diffusion coefficient
L = 10
S = 10
K = 10

# Temperatures at A and B
Ta = 0
Tb = 100

# Number of control volumes
N = int(10)

'''
--------------------------------------------------------------------------------
                               CONSTRUCT MESH
--------------------------------------------------------------------------------
'''

# Control volume length
dx = L/N

# Outward pointing normal at west and east face
ne = 1
nw = -1

# Cell index
c = np.linspace(0, N-1 , N, dtype=int)

# Face centre locations
xf = np.linspace(0, L, N+1)

# Cell centre locations
xc = np.linspace((0+dx/2), (L-dx/2), N)

'''
--------------------------------------------------------------------------------
                         FACES INTERPOLATION STENCIL
--------------------------------------------------------------------------------
'''

# Construct interpolation stencil for each face
Nn_west = 3
Nn_east = 3

faceCellNei = [[] for x in range(N + 1)]

# Loop for computational stencil construction
# Each cell has the same number of neighbours
Nn = Nn_east + Nn_west

for i, stencil in enumerate(faceCellNei):

    # Interior stencils
    if i > (Nn_west - 1) and i < (N + 1 - Nn_east):

        for k in range(0, Nn_west):
            stencil.append(i - k - 1)

        for k in range(0, Nn_east):
            stencil.append(i + k)

    # Boundary stencil is for one cell smaller in size becouse boundary face
    # is also going in stencil

    # Modified boundary stencils on left side
    if i < Nn_west:
        if i != 0:
            for k in range(Nn - i):
                stencil.append(i + k)
            for k in range(Nn - len(stencil)):
                stencil.append(i - k - 1)
        else:
            for k in range(Nn - i - 1):
                stencil.append(i + k)

    # Modified boundary stencils on right side
    if i > (N - Nn_east):
        if i != N:
            for k in range(i, N):
                stencil.append(k)
            for k in range(Nn - len(stencil)):
                stencil.append(i - k - 1)
        else:
            for k in range(Nn - len(stencil) - 1):
                stencil.append(i - k - 1)

    stencil.sort()

    print(f'Interpolation stencil for face [{i}] is: {stencil}')

'''
--------------------------------------------------------------------------------
                             WEIGHT FUNCTION
--------------------------------------------------------------------------------
'''
def weight(dist) -> float:

     # Shape parameter of the kernel (constant)
     k = 6

     # Smoothing length
     dm = 2 * (max(Nn_east, Nn_west) * dx - dx * 0.5)

     w = ( np.exp(-pow(dist / dm, 2) * pow(k, 2)) - np.exp(-pow(k, 2)) ) \
         / (1 - np.exp(-pow(k, 2)) )
     return w

'''
--------------------------------------------------------------------------------
                         LOCAL REGRESSION ESTIMATORS
--------------------------------------------------------------------------------
'''

# Number of terms in Taylor expansion
Np = 4

# Coefficient for each face interpolation molecule
cx = [[] for x in range(N + 1)]

# Loop over all interior faces
for facei in range(0, len(xf)):
    print(f'\033[1m\nConstructing weights for face {facei}\033[0m')

    # ((rows, cols))
    Q = np.zeros((Np, Nn))
    W = np.zeros((Nn, Nn))

    # Get number of cell centre neighbours (excluding ghost cells)
    NnCells = len(faceCellNei[facei])

    # Loop over neighbours Nn and construct matrix Q
    # (n-th column of Q is q(xn-x))
    for i in range(NnCells):
        for n in range(Np):
            Q[n, i] = pow((xc[faceCellNei[facei][i]] - xf[facei]), n) \
                      * np.math.factorial(n)

    #for i in range(NnCells):
    #    Q[:, i] = [1, (xc[faceCellNei[facei][i]] - xf[facei])]

    # Loop over neighbours Nn and construct matrix W (diagonal matrix)
    w_diag = np.zeros(Nn)
    for i in range(NnCells):
        dist = xc[faceCellNei[facei][i]] - xf[facei]
        w_diag[i] = weight(abs(dist))

    # Check if stencil has ghost face and modify W and Q in that case
    # Add unit weight for boundary face in W
    # Add truncated Taylor series with zero terms at any location except first
    if (NnCells < Nn):
        w_diag[Nn - 1] = 1.0
        Q_bc = np.zeros(Np)
        Q_bc[0] = 1.0
        Q[:, Nn - 1] = Q_bc

    diagonal_indices = np.diag_indices(W.shape[0])
    W[diagonal_indices] = w_diag

    # Calculate matrix M = Q*W*Q^T
    M = Q @ W @ Q.T

    if (M.shape[0] != M.shape[1] | M.shape[0] == Np):
        raise ValueError("Matrix M should be Np x Np shape")

    # Calculate matrix A = M^-1*Q*W
    A = np.linalg.inv(M) @ Q @ W

    # Second row of matrix A are coefficients that are multiplying cell value
    # to obtain face gradient
    cx[facei] = list(A[1, :])

    print(f'\033[94mMatrix Q:\n{Q} \033[0m')
    print(f'\033[92mMatrix W:\n{W} \033[0m')
    print(f'\033[93mMatrix M:\n{M} \033[0m')
    print(f'\033[95mMatrix A:\n{A} \033[0m')
    print(f'\033[96mVector coeffs cx:\n{cx[facei]} \033[0m')

'''
--------------------------------------------------------------------------------
                         ASSEMBLE SYSTEM OF EQUATIONS
--------------------------------------------------------------------------------
'''

# Initialise temperature vector
T = np.zeros((N, 1))

# Initialise source vector
b = np.zeros((N, 1))

# Initialise matrix of coeffs
A = np.zeros((N, N))

# Treat interior cell
for celli in range(0, N):

    # Left(west) face stencil
    leftFaceIndex = celli
    for i in range(len(faceCellNei[leftFaceIndex])):
        nCellIndex = faceCellNei[leftFaceIndex][i]
        A[celli][nCellIndex] += K * S * nw * cx[leftFaceIndex][i]

    # Right(east) face stencil
    rightFaceIndex = celli + 1
    for i in range(len(faceCellNei[rightFaceIndex])):
        nCellIndex = faceCellNei[rightFaceIndex][i]
        A[celli][nCellIndex] += K * S * ne * cx[rightFaceIndex][i]

    if(celli == 0):
        b[celli] += - K * S * nw * Ta * cx[0][Nn-1]

    if(celli == N-1):
        b[celli] += - K * S * ne * Tb * cx[N][Nn - 1]

print('\n', A,'\n')
print(b,'\n')

'''
--------------------------------------------------------------------------------
                           SOLVE SYSTEM OF EQUATIONS
--------------------------------------------------------------------------------
'''

# Solve system of equations
T = np.linalg.solve(A,b)
T = np.concatenate(T).tolist()

'''
--------------------------------------------------------------------------------
                                 PLOT RESULTS
--------------------------------------------------------------------------------
'''

# Analytical solution
# Assumed analytical manufactured solution
xa = np.linspace(0,L,100)
Ta = np.sin(2.0*np.pi*pow(xa,2)/100.0)

Ta = np.linspace(0,Tb,11)
Le = np.linspace(0,L,11)

plt.figure()
plt.plot( Le, Ta, '-', xc, T, '-o', markersize = 4)

plt.title('Temperature distribution', fontsize = 12, weight = 600)
plt.xlabel('x', fontsize = 12, weight = 600)
plt.ylabel('Temperature', fontsize = 12, weight = 600)
plt.legend(['analytical', 'numerical'])
plt.show()

avgRelError = 0.0
maxRelError = 0.0
for i, k in enumerate(T):
    analytical = ( xc[i] / L ) * Tb
    relError = abs((k - analytical) / analytical)
    relError *= 100
    avgRelError += relError
    if(relError > maxRelError):
        maxRelError = relError

avgRelError /= N

print(f'Average relative error in %: {avgRelError}')
print(f'Maximal relative error in %: {maxRelError}')
