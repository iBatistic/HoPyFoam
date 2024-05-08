"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Field class as base clas for volume and surface field
"""

import re
import numpy as np

class field():

    _fieldName = "None"

    def __init__(self, fieldName="None"):
        self._fieldName = fieldName

    # foamFile banner
    def foamFile(self, fieldType, fieldName, timeValue):
        banner = 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
        banner += '    class       ' + fieldType + ';\n'
        banner += '    location    "' + str(timeValue) + '";\n'
        banner += '    object     ' + fieldName + ';\n}\n'
        return banner