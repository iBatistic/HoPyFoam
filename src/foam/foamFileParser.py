"""
 _____     _____     _____ _____ _____      |  Python code for High-order FVM
|  |  |___|  _  |_ _|   __|  |  |     |     |  Python Version: 3.10
|     | . |   __| | |   __|  |  | | | |     |  Code Version: 0.0
|__|__|___|__|  |_  |__|   \___/|_|_|_|     |  License: GPLv3
                |___|

Description
    Collection of parser functions for OpenFOAM files
"""
import re
import numpy as np

POLYMESH = 'constant/polyMesh'

def read_points_file() -> np.ndarray:

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
    return points

def read_faces_file() -> np.ndarray:

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
    return faces

def read_owner_file() -> np.ndarray:

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
    return owner

def read_neighbour_file()  -> np.ndarray:

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
    return neighbour

def read_boundary_File() -> dict:

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

    return boundary

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

def convert_to_int(string):
    if np.char.isdigit(string):
        return int(string)
    else:
        return string