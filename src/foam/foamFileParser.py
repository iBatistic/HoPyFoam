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
import numpy as np

POLYMESH = 'constant/polyMesh'
SYSTEM = 'system'
ZERO = '0'

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

            for i in range(1,len(file),3):
                index = int(i/3)
                points[index][0] = file[i]
                points[index][1] = file[i+1]
                points[index][2] = file[i+2]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/points not found!")
    except Exception as e:
        print(f"An error occured: {e}")

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
            faces = np.empty(nFaces, dtype = object)

            faceCounter = 0
            for i in range(nFaces):
                faceSize = int(file[faceCounter])
                face = np.empty(faceSize, int)
                for j in range(int(file[faceCounter])):
                    face[j] = file[faceCounter+j+1]
                faces[i] = face
                faceCounter += faceSize + 1

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/faces not found!")
    except Exception as e:
        print(f"An error occured: {e}")

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

            for i in range(1,len(file),1):
                owner[i-1] = file[i]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/owner not found!")
    except Exception as e:
        print(f"An error occured: {e}")

    return owner

def read_neighbour_file()  -> np.ndarray:
    try:
        with open(POLYMESH + '/neighbour') as f:
            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r'[()]', '', file)

            file = file.split()
            nNeighbour = int(file[0])
            neighbour = np.empty(nNeighbour, dtype=int)

            for i in range(1,len(file),1):
                neighbour[i-1] = file[i]

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/neighbour not found!")
    except Exception as e:
        print(f"An error occured: {e}")

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
                for j in range(0,len(patchDictList),2):
                    patchData.update(
                            {patchDictList[j]: convert_to_int(patchDictList[j+1])}
                        )

                boundary[file[i]] = patchData
    except FileNotFoundError:
        print(f"Error: {POLYMESH}/boundary not found!")
    except Exception as e:
        print(f"An error occured: {e}")

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

                # Read line only if it has two enties, otherwise something probably
                # went wrong
                if len(line) != 2 or line[0] == '' or line[1] == '':
                    raise TypeError("Something went wrong when parsing controlDict")
                else:
                    controlDict.update({line[0]: convert_to_float(line[1])})

    except FileNotFoundError:
        print(f"Error: {POLYMESH}/controlDict not found!")
    except Exception as e:
        print(f"An error occured: {e}")
    return controlDict

def readBoundaryAndInitialConditions(fileName) -> tuple[str, list, dict]:
    boundaryConditionsDict = dict()
    dataType, initialValue = '', ''
    boundaryConditions = []

    try:
        with (open(ZERO+ '/' + fileName) as f):

            file = f.read()
            file = remove_cpp_comments(file)
            file = remove_foamFile_dict(file)
            file = re.sub(r';', '', file)
            file = [line for line in file.splitlines() if line.strip()]

            # Remove dimensions line and internalField(initial field value) line, pass boundaryField line
            for line in file:
                if line.startswith('internalField'):
                    initialValue = line.strip().split()[2]
                elif line.startswith('dimensions'):
                    dataType = line.strip()
                elif line.startswith('boundaryField'):
                    pass
                else:
                    boundaryConditions.append(line)

            file = ''.join(boundaryConditions)
            file = re.sub(r'^\{(.*)\}$', r'\1', file)

            # Extract data inside curly brackets
            patchDicts = re.findall(r'\{([^}]+)\}', file)

            # Extract patch name above curly brackets
            patchNames = re.sub(r'{[^}]*}*', ' ', file)
            patchNames = patchNames.split()

            for i in range(len(patchNames)):
                patchDict = patchDicts[i].split()

                if(patchDict[1] == 'empty'):
                    boundaryConditionsDict.update({patchNames[i]: {"type": patchDict[1]}})
                else:
                    boundaryConditionsDict.update({patchNames[i]: \
                                                       {"type": patchDict[1], patchDict[2]: {patchDict[3]: patchDict[4]} }})
    except FileNotFoundError:
        print(f"Error: {ZERO}/{fileName} not found!")
    except Exception as e:
        print(f"An error occured: {e}")

    # Clean dataType from brackets and put dimensions in list
    dataType = dataType.split()[1:]
    for index, element in enumerate(dataType):
        dataType[index]= re.sub(r'\D', '', element)

    return initialValue, dataType, boundaryConditionsDict

#https://stackoverflow.com/questions/241327/remove-c-and-c-comments-using-python
def remove_cpp_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
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


def convert_to_float(string, check=False):
    if (check):
        if np.char.isdigit(string):
            return float(string)
        else:
            raise ValueError("Float digit expected")

    if np.char.isdigit(string):
        return float(string)
    else:
        return string