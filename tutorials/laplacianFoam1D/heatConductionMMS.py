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

# Temperatures at left side (Ta) and right side (Tb)
Ta = 0
Tb = 0

# Number of control volumes
N = int(16)

# Number of terms in Taylor expansion
Np = 4

# Number of face west and east neighbours
Nn_west = 4
Nn_east = 4

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

# Construct empty interpolation stencil for each face
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

    # Boundary stencil is for one cell smaller in size because boundary face
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
                             SOURCE TERM
--------------------------------------------------------------------------------
'''
# Volume integral of the source term from x = a to x = b, where b > a
def sourceTermVolumeIntegral(a, b) -> float:

    # Calculate the integral where we assume y = z = 1
    return (1.0/25.0)*(np.pi*b*np.cos((np.pi*pow(b,2))/50)
                       - np.pi*a*np.cos((np.pi*pow(a,2))/50))

'''
--------------------------------------------------------------------------------
                             WEIGHT FUNCTION
--------------------------------------------------------------------------------
'''
def weight(dist) -> float:

     # Shape parameter of the kernel (constant)
     k = 6

     # Smoothing lengt
     dm = 2 * (max(Nn_east, Nn_west) * dx - dx * 0.5)

     w = ( np.exp(-pow(dist / dm, 2) * pow(k, 2)) - np.exp(-pow(k, 2)) ) \
         / (1 - np.exp(-pow(k, 2)) )
     return w

'''
--------------------------------------------------------------------------------
                         LOCAL REGRESSION ESTIMATORS
--------------------------------------------------------------------------------
'''
# Check number of neighbours and order of Taylor series
if (Np > Nn):
    raise ValueError(f'Number of neighbours {Nn} is smaller than length of '
                     f'Taylor expansion {Np}')

# Coefficient for each face interpolation molecule
cx = [[] for x in range(N + 1)]

# Loop over all faces
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
        # Loop over elements in truncated Taylor series and add their value
        for n in range(Np):
            Q[n, i] = pow((xc[faceCellNei[facei][i]] - xf[facei]), n) / np.math.factorial(n)

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
    print(f'\033[96mVector gradCoeffs cx:\n{cx[facei]} \033[0m')

'''
--------------------------------------------------------------------------------
                         ASSEMBLE SYSTEM OF EQUATIONS
--------------------------------------------------------------------------------
'''

# Initialise temperature vector
T = np.zeros((N, 1))

# Initialise source vector
b = np.zeros((N, 1))

# Initialise matrix of gradCoeffs
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

    # Contribution to source from the first cell boundary face
    if(celli == 0):
        b[celli] += - K * S * nw * Ta * cx[0][Nn-1]

    # Contribution to source from the last cell boundary face
    if(celli == N-1):
        b[celli] += - K * S * ne * Tb * cx[N][Nn - 1]

    # Add source term
    for celli in range(0, N):
        # Add the source term using the west and east face centre coordinates
        b[celli] +=  dx * S * sourceTermVolumeIntegral(xf[celli], xf[celli + 1])

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

plt.figure()
plt.plot( xa, Ta, '-', xc, T, 'o', markersize = 4)

plt.title('Temperature distribution', fontsize = 12, weight = 600)
plt.xlabel('x', fontsize = 12, weight = 600)
plt.ylabel('Temperature', fontsize = 12, weight = 600)
plt.legend(['analytical', 'numerical'])
plt.show()

# Relative error, L_2 and L_infinity norm
avgRelError = 0.0
maxRelError = 0.0
L2= 0.0
Linf = 0.0
for i, k in enumerate(T):
    analytical = np.sin(2.0*np.pi*pow(xc[i],2)/100.0)
    diff = abs(k - analytical)

    relError = (abs(k - analytical) / abs(analytical))*100

    avgRelError += relError
    if(relError > maxRelError):
        maxRelError = relError

    L2 += np.math.pow(diff, 2)

    if(diff > Linf):
        Linf = diff

    # print(f'Cell {i} relative error: {relError:.3f}')

L2 = np.math.sqrt(L2/N)
avgRelError /= N

print(f'\nAverage relative error in %: {avgRelError:.3f}')
print(f'Maximal relative error in %: {maxRelError:.3f}')
print(f'L_2 error: {L2:.10f}')
print(f'L_infinty error: {Linf:.10f}')


mesh_size = [10, 100, 1000]

L_2_p5 = [0.0152759845, 0.0000053969,  0.0000000004]
L_inf_p5 = [0.0310252646, 0.0000106041, 0.0000000009]

L_2_p4 = [0.011, 0.0000026625, 0.0000000002]
L_inf_p4 = [0.035, 0.0000054295, 0.0000000004]

L_2_p3 = [0.0499618835, 0.0002073746, 0.0000020949]
L_inf_p3 = [0.0816599990, 0.0003646903, 0.0000046447]

L_2_p2 = [ 0.0277484962, 0.0002592340, 0.0000025907]
L_inf_p2 = [0.0592752853, 0.0006217822, 0.0000062167]

def order(np, order, offset):
    result = []
    for i in range(len(np)):
        result.append(offset*(1/np[i])**order)
    return result

L_2 = order(mesh_size, 2, 5)
L_3 = order(mesh_size, 3, 100)
L_4 = order(mesh_size, 4, 100)
L_5 = order(mesh_size, 5, 5000)

L_inf_2 = order(mesh_size, 2, 10)
L_inf_3 = order(mesh_size, 3, 100)
L_inf_4 = order(mesh_size, 4, 100)
L_inf_5 = order(mesh_size, 5, 5000)
print(L_2)

plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$n_b$', fontsize = 12, weight = 600)
plt.title('$L_{2}$ norm convergence log-log', fontsize = 12, weight = 600)
plt.ylabel('$L_{2}$', fontsize = 12, weight = 600)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.plot( mesh_size, L_2_p5, '-o', mesh_size, L_2_p4, '-o', mesh_size, L_2_p3, '-o', mesh_size, L_2_p2, '-o', markersize = 2)
plt.plot( mesh_size, L_2, '--', color = "red", linewidth = 1)
plt.plot( mesh_size, L_3, '--', color = 'green', linewidth = 1)
plt.plot( mesh_size, L_4, '--', color = 'orange',  linewidth = 1)
plt.plot( mesh_size, L_5, '--', color = 'blue',  linewidth = 1)
plt.legend(['5 order','4 order', '3 order', '2 order'])
plt.savefig('L2.png', bbox_inches='tight')
plt.show()


plt.figure()
plt.yscale('log')
plt.xscale('log')
plt.title('$L_{\infty}$ norm convergence log-log', fontsize = 12, weight = 600)
plt.xlabel('$n_b$', fontsize = 12, weight = 600)
plt.ylabel('$L_{\infty}$', fontsize = 12, weight = 600)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.plot( mesh_size, L_inf_p5, '-o', mesh_size, L_inf_p4, '-o', mesh_size, L_inf_p3, '-o', mesh_size, L_inf_p2, '-o', markersize = 2)
#plt.plot( mesh_size, L_inf_2, '--', color = "red", linewidth = 1)
#plt.plot( mesh_size, L_inf_3, '--', color = 'green', linewidth = 1)
#plt.plot( mesh_size, L_inf_4, '--', color = 'orange',  linewidth = 1)
#plt.plot( mesh_size, L_inf_5, '--', color = 'blue',  linewidth = 1)
plt.legend(['5 order','4 order', '3 order', '2 order'])
plt.savefig('Linfty.png', bbox_inches='tight')
plt.show()

sol_2 = [ -9.099325166253853e-05, 0.1251427577696729, 0.36839227073774666, 0.6865144131917624, 0.9556683503208502, 0.9554863638175252, 0.474698011694269, -0.40339413748168007, -1.0443846114782465, -0.6282275374662959]
#sol_4 = [12.54157, 11.80158, 7.48726, -4.89682,-26.93359,-48.06064 ,-39.73038, 23.71017 ,105.71475, 84.0298]
print(T)
plt.figure()
plt.title('Solution on mesh with 10 CVs (each face has 10 neighbours)', fontsize = 12, weight = 600)
plt.xlabel('$x$', fontsize = 12, weight = 600)
plt.ylabel('$T$', fontsize = 12, weight = 600)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.plot( xa, Ta, '-',color = "black", linewidth = 0.5)
plt.plot( xc, T, 'o', color = "blue",  markersize = 3)
#plt.plot( xc, sol_2, 'o', color = "red", linewidth = 1, markersize = 3)
plt.legend(['analytical','Np = 10','2 order'])
plt.savefig('4vs2order.png', bbox_inches='tight')
plt.show()
