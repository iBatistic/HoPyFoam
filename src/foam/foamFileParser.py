"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Collection of parser functions for OpenFOAM files
"""
import re
import sys
import warnings

import numpy as np

CONSTANT = 'constant/'
POLYMESH = CONSTANT + 'polyMesh'
MECHANICALPROPERTIES = CONSTANT + 'mechanicalProperties'
SYSTEM = 'system'
ZERO = '0'

SWICHVALUES = {
    "no": False,
    "off": False,
    "yes": True,
    "on": True,
}
def convertToBool(string):
    return SWICHVALUES.get(string.lower(), None)

def read_points_file() -> np.ndarray:
    try:
        with open(POLYMESH + '/points') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', '', file)

            file = file.split()
            nPoints = int(file[0])
            points = np.empty([nPoints, 3], dtype=float)

            for i in range(1, len(file), 3):
                index = int(i / 3)
                points[index][0] = file[i]
                points[index][1] = file[i + 1]
                points[index][2] = file[i + 2]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/points not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return points


def read_faces_file() -> np.ndarray:
    try:
        with open(POLYMESH + '/faces') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', ' ', file)

            file = file.split()
            nFaces = int(file.pop(0))
            faces = np.empty(nFaces, dtype=object)

            faceCounter = 0
            for i in range(nFaces):
                faceSize = int(file[faceCounter])
                face = np.empty(faceSize, int)
                for j in range(int(file[faceCounter])):
                    face[j] = file[faceCounter + j + 1]
                faces[i] = face
                faceCounter += faceSize + 1

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/faces not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return faces


def read_owner_file() -> np.ndarray:
    try:
        with open(POLYMESH + '/owner') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', '', file)

            file = file.split()
            nOwner = int(file[0])
            owner = np.empty(nOwner, dtype=int)

            for i in range(1, len(file), 1):
                owner[i - 1] = file[i]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/owner not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return owner


def read_neighbour_file() -> np.ndarray:
    try:
        with open(POLYMESH + '/neighbour') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', '', file)

            file = file.split()
            nNeighbour = int(file[0])
            neighbour = np.empty(nNeighbour, dtype=int)

            for i in range(1, len(file), 1):
                neighbour[i - 1] = file[i]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/neighbour not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return neighbour


def read_boundary_File() -> dict:
    try:
        with open(POLYMESH + '/boundary') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', '', file)
            file = re.sub(r';', '', file)

            # Extract data inside curly brackets
            patchDicts = re.findall(r'\{([^}]+)\}', file)

            # Extract patch name above curly brackets
            file = re.sub(r'{[^}]*}*', '', file)
            file = file.split()

            nPatches = int(file.pop(0))
            boundary = dict()

            for i in range(nPatches):
                patchDictList = patchDicts[i].split()
                patchData = dict()
                for j in range(0, len(patchDictList), 2):
                    patchData.update(
                        {patchDictList[j]: convert_to_int(patchDictList[j + 1])}
                    )

                boundary[file[i]] = patchData
    except FileNotFoundError:
        print(f"Error: {POLYMESH}/boundary not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return boundary


def read_controlDict_file() -> dict:
    controlDict = dict()
    try:
        with open(SYSTEM + '/controlDict') as f:

            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)

            file = [line for line in file.splitlines() if line.strip()]

            for i in file:
                # Split line using one or more spaces
                line = re.split(r'\s+', i)

                # Read line only if it has two entries, otherwise something probably
                # went wrong
                if len(line) != 2 or line[0] == '' or line[1] == '':
                    raise TypeError("Something went wrong when parsing controlDict")
                else:
                    controlDict.update({line[0]: convert_to_float(line[1])})

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/controlDict not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return controlDict

def readTransportProperties(name) -> tuple[float, list]:
    print(f"Reading diffusivity {name} in transportProperties dict\n")

    try:
        with (open(CONSTANT + 'transportProperties') as f):

            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)
            file = [line for line in file.splitlines() if line.strip()]

            for line in file:
                # Split line using one or more spaces
                line = re.split(r'\s+', line)
                gamma = convert_to_float(line[-1], True)

            # Clean dataType from brackets and put dimensions in list
            dataType = line[2:-1]
            for index, element in enumerate(dataType):
                dataType[index] = re.sub(r'\D', '', element)

            # Remove empty strings from list
            dataType = list(filter(None, dataType))

    except FileNotFoundError:
        print(f"Error: {CONSTANT}/transportProperties' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    return dataType, gamma


# https://stackoverflow.com/questions/241327/remove-c-and-c-comments-using-python
def remove_cpp_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def remove_foamFile_dict(text):
    # Remove FoamFile string
    text = re.sub(r'FoamFile', '', text)
    # Remove everything inside curly brackets after FoamFile string
    text = re.sub(r'{[^}]*}*', '', text, 1)
    return text


def convert_to_int(string, check=False):
    if (check):
        if np.char.isdigit(string):
            return int(string)
        else:
            raise ValueError("Integer digit expected")

    if np.char.isdigit(string):
        return int(string)
    else:
        return string


def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def convert_to_float(string, check=False):
    if check:
        if isFloat(string):
            return float(string)
        else:
            raise ValueError("Float digit expected")

    if isFloat(string):
        return float(string)
    else:
        return string

def readMechanicalProperties(path=MECHANICALPROPERTIES, convertToLameCoeffs=True) -> tuple[float, float]:

    print(f"Reading mechanicalProperties dict\n")

    try:
        with (open(path) as f):
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)
            file = [line for line in file.splitlines() if line.strip()]
            planeStress = convertToBool(file[0].split()[1])

            file = ''.join(file)
            file = re.sub(r'^\{(.*)\}$', r'\1', file)

            # Extract data inside curly brackets
            properties = re.findall(r'\{([^}]+)\}', file)
            properties = properties[0].split()

            E = convert_to_float(properties[properties.index('E')+9])
            nu = convert_to_float(properties[properties.index('nu')+9])
    except SyntaxError as e:
        print(f'Syntax error: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {path} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    if not convertToLameCoeffs:
        print(f"E: {E:.5f}\n, nu: {nu:.5f}")
        return E, nu

    mu = E / (2.0*(1.0 + nu))

    if nu > (0.499):
        print(f"second Lame\'s parameter mu: {mu:.5f}\n"
              f"First Lame\'s parameter lambda: None")
        return mu, None

    if planeStress:
        lam = nu * E / ((1.0 + nu)*(1.0 - nu))
    else:
        lam = nu * E / ((1.0 + nu)*(1.0 - 2.0*nu))

    print(f'First Lame\'s parameter lambda: {lam:.5f}, '
          f'second Lame\'s parameter mu: {mu:.5f}\n')

    return mu, lam

def readScalarField(fileName, time=ZERO) -> tuple[str, list, dict]:
    boundaryConditionsDict = dict()
    dataType, initialValue = '', ''
    data = []
    cellValues = [[]]

    print(f"Reading field {fileName}\n")

    try:
        with (open(time + '/' + fileName) as f):

            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)
            file = [line for line in file.splitlines() if line.strip()]
            
            # Remove dimensions line and internalField(initial field value) line, pass boundaryField line
            for line in file:
                if line.startswith('internalField'):
                    initialValue = line.strip().split()[-1]
                    valueType = line.strip().split()[1]
                elif line.startswith('dimensions'):
                    dataType = line.strip()
                elif line.startswith('boundaryField'):
                    pass
                else:
                    data.append(line)
            
            # This part of code reads nonuniform cell field
            if (valueType == 'nonuniform'):

                firstIndex = data.index('(') + 1
                lastIndex = data.index(')')

                values = data[firstIndex:lastIndex]
                
                for index, cellVal in enumerate(values):
                    cellValues.append([convert_to_float(cellVal)])
                # Just clean empty list which is constructed on initialisation
                cellValues = [sublist for sublist in cellValues if sublist]
                
                del data[firstIndex:lastIndex]

                numberOfCells = data.pop(0)
                if(len(cellValues) != convert_to_int(numberOfCells)):
                    raise SyntaxError(f'Number of elements {len(cellValues)} does not match'
                                      f' required number {convert_to_int(numberOfCells)}')
                # Remove ( and ) brackets inside which cell data was stored
                del data[0:2]
                
            elif (valueType == 'uniform'):
                cellValues[0] = ([convert_to_float(initialValue)])
            else:
                raise SyntaxError(f'Value type should be unifrom or nonuniform, type is {dataType}')

            # Read boundary conditions
            file = ''.join(data)
            file = re.sub(r'^\{(.*)\}$', r'\1', file)

            # Extract data inside curly brackets
            patchDicts = re.findall(r'\{([^}]+)\}', file)

            # Extract patch name above curly brackets
            patchNames = re.sub(r'{[^}]*}*', ' ', file)
            patchNames = patchNames.split()
            
            for i in range(len(patchNames)):
                patchDict = patchDicts[i].split()
                if (patchDict[1] == 'empty' or patchDict[1] == 'zeroGradient'):
                    boundaryConditionsDict.update({patchNames[i]: {"type": patchDict[1]}})
                else:
                    value = convert_to_float(patchDict[4])
                    boundaryConditionsDict.update({patchNames[i]: \
                                                       {"type": patchDict[1],
                                                       patchDict[2]: {patchDict[3]: value}}})
    except SyntaxError as e:
        print(f'Syntax error: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {time}/{fileName} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)
    
    # Clean dataType from brackets and put dimensions in list
    dataType = dataType.split()[1:]
    for index, element in enumerate(dataType):
        dataType[index] = re.sub(r'\D', '', element)
    
    return cellValues, dataType, boundaryConditionsDict

def readVectorField(fileName, time=ZERO) -> tuple[str, list, dict]:
    boundaryConditionsDict = dict()
    dataType, initialValue = '', []
    data = []
    cellValues = [[]]

    print(f"Reading field {fileName}\n")

    try:
        with (open(time + '/' + fileName) as f):

            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)
            file = [line for line in file.splitlines() if line.strip()]

            # Remove dimensions line and internalField(initial field value) line, pass boundaryField line
            for line in file:
                if line.startswith('internalField'):
                    valueType = line.strip().split()[1]
                    if(valueType == 'uniform'):
                        initialValue = [float(s.strip('()')) for s in line.strip().split()[-3:]]

                elif line.startswith('dimensions'):
                    dataType = line.strip()
                elif line.startswith('boundaryField'):
                    pass
                else:
                    data.append(line)

            # This part of code reads nonuniform cell field
            if valueType == 'nonuniform':
                fieldSize = int(file[2])
                cellValues =  [np.zeros(3) for _ in range(fieldSize)]
                for i in range(fieldSize):
                    vector = file[4 + i].strip('( )').split()
                    cellValues[i] = np.array([float(vector[0]), float(vector[1]), float(vector[2])])
                file = file[4+fieldSize+1:]
            elif valueType == 'uniform':
                cellValues[0] = initialValue
            else:
                raise SyntaxError(f'Value type should be unifrom or nonuniform, type is {dataType}')

            if (valueType == 'nonuniform'):
                warnings.warn(f"Boundary condition reader not implemented "
                            f" for non-uniform boundary fields. Value from 0 will be"
                            f" taken", stacklevel=3)

                boundaryConditionsDict = readVectorField('U', '0')[2]

            else:
                # Read boundary conditions
                file = ''.join(data)
                file = re.sub(r'^\{(.*)\}$', r'\1', file)

                # Extract data inside curly brackets
                patchDicts = re.findall(r'\{([^}]+)\}', file)

                # Extract patch name above curly brackets
                patchNames = re.sub(r'{[^}]*}*', ' ', file)
                patchNames = patchNames.split()

                for i in range(len(patchNames)):
                    patchDict = patchDicts[i].split()

                    if (patchDict[1] == 'empty'):
                        boundaryConditionsDict.update({patchNames[i]: {"type": patchDict[1]}})
                    else:
                        #pass
                        value = [float(s.strip('()')) for s in patchDict[4:]]
                        boundaryConditionsDict.update({patchNames[i]: \
                                                           {"type": patchDict[1],
                                                           patchDict[2]: {patchDict[3]: value}}})

    except SyntaxError as e:
        print(f'Syntax error: {e}')
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {time}/{fileName} not found!")
        sys.exit(1)
    except Exception as e:
        print(f"An error occured: {e}")
        sys.exit(1)

    # Clean dataType from brackets and put dimensions in list
    dataType = dataType.split()[1:]
    for index, element in enumerate(dataType):
        dataType[index] = re.sub(r'\D', '', element)

    return cellValues, dataType, boundaryConditionsDict
