"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Class for convergence informations and convergence checks of the
    solution loop.
"""

from src.foam.foamFileParser import *

class solutionControl():

    def __init__(self):
        self._controlDict = read_controlDict_file()
        self._time = 0.0

        # Store main entries which each controlDict has
        self._endTime = self._controlDict.get('endTime')
        self._startTime = self._controlDict.get('startTime')
        self._deltaT = self._controlDict.get('deltaT')
        self._startFrom = self._controlDict.get('startFrom')
        self._stopAt = self._controlDict.get('stopAt')

        if(self._startFrom == 'startTime'):
            self._time = self._startTime

    def controlDict(self):
        return self._controlDict

    def time(self) -> float:
        return self._time

    def loop(self) -> bool:

        if(self._time < self._endTime):
            self._time += self._deltaT
            return True
        else:
            return False
