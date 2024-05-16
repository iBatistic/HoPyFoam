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
from src.foam.foamFileParser import *

class LaplacianOperator():

    @classmethod
    def Laplacian(self, psi, gamma):

        print('Assembling system of equations for Laplacian operator\n')

        mesh = psi._mesh
        nCells = mesh.nCells()
        nFaces = mesh.nFaces()
        nInternalFaces = mesh.nInternalFaces()
        dimensions = psi._dimensions

        source = np.zeros([nCells, dimensions], dtype=float)

        A = np.zeros([nCells, nCells], dtype=float)

        owner = mesh._owner
        neighbour = mesh._neighbour

        GaussPointsAndWeights = psi._facesGaussPointsAndWeights

        # Loop over faces
        for cmpt in range(dimensions):
            for faceI in range(nFaces):

                # Diffusivity coefficient multiplied with face magnitude
                gammaMagSf = mesh.magSf()[faceI] * gamma

                # Face normal
                nf = mesh.nf()[faceI]

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

                        # Internal face treatment
                        if (faceI < nInternalFaces):
                            # Owner and neighbour of current face
                            cellP = owner[faceI]
                            cellN = neighbour[faceI]

                            # Store Laplace coefficients
                            A[cellP][cellIndex] += gammaMagSf * gpW * (cx[j] @ nf)
                            A[cellN][cellIndex] += - gammaMagSf * gpW * (cx[j] @ nf)

                        else:
                            # Boundary face treatment
                            cellP = owner[faceI]
                            # Ovo ne moze tu bit za zeroGradient!!!!
                            # Napraviti kako je u foamu, ic po patchevima i onda za svaki patch
                            # Dodat kontribuciju.
                            A[cellP][cellIndex] += gammaMagSf * gpW * (cx[j] @ nf)

                            if (j == len(faceStencil) - 1):
                                # Boundary face centre is not included in face stencil list
                                # Its contribution is added last, after cell centres

                                patchType = mesh.facePatchType(faceI, psi)
                                patchName = mesh.facePatchName(faceI, psi)
                                addSourceContribution = eval("self." + patchType)

                                # This can be done in more elegant manner
                                addSourceContribution(self, gp, faceI, psi, patchName, source, cellP, gammaMagSf, nf, gpW,
                                                      cx[j + 1])

                        # Face have same contribution for cellP and cellN
                        # print(f'face owner: {cellP}')
                        # print(f'face neighbour: {cellN}')
                        # print(f'face normal: {nf}')
                        # print(f'i counter, cellIndex: {i}, {cellIndex}')
                        # print(f'face Gauss point interpolation coeffcitient: cx')
                        # print(f'face centre: {mesh.Cf()[faceI]}')
                        # print(f'stencil neigbour: {cellIndex}')

        # File to write data for debug
        fileName = "LaplacianMatrix.debug.txt"

        # Remove old file
        if os.path.exists(fileName):
            os.remove(fileName)

        # Write matrix and solution vector
        file = open(fileName, 'w')
        file.write(f'{str(A)} \n {str(source)}')
        file.close()

        return source, A

    def fixedValue(self, gp, faceI, psi, patchName, source, cellP, gammaMagSf, nf, gpW, cx):

        value = convert_to_float(psi._boundaryConditionsDict[patchName]['value']['uniform'])
        source[cellP][0] += -gammaMagSf * gpW * (cx @ nf) * value

    def analyticalFixedValue(self, gp, faceI, psi, patchName, source, cellP, gammaMagSf, nf, gpW, cx):

        # TO_DO ovisno o gausovoj tocki promjenit iznos!

        # I do like CFD, page 222, equation b
        # value depends on Gauss point location
        faceCentre = psi._mesh.Cf()[faceI]
        x = gp[0]#faceCentre[0]
        y = gp[1]#faceCentre[1]
        #x = faceCentre[0]
        #y = faceCentre[1]
        # print(f'x: {x}, y: {y}')
        #value = x * x - y * y
        # value = 2.0*y / (pow((1+x),2)+pow(y,2))
        value = np.sin(1*y) * np.exp(1*x)

        source[cellP][0] += -gammaMagSf * gpW * (cx @ nf) * value

    def empty(self, gp, faceI, psi, patchName, source, cellP, gammaMagSf, nf, gpW, cx):
        pass

    def zeroGradient(self, gp, faceI, psi, patchName, source, cellP, gammaMagSf, nf, gpW, cx):
        pass