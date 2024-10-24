"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Base clas for volume and surface field
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['field']

class field():

    def __init__(self, fieldName="None"):
        self._fieldName = fieldName

    @property
    def fieldName(self):
        return self._fieldName

    # foamFile banner. Used when writing fields in OF format.
    def foamFileDict(self, fieldType, fieldName, timeValue) -> str:
        banner = 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
        banner += '    class       ' + fieldType + ';\n'
        banner += '    location    "' + str(timeValue) + '";\n'
        banner += '    object     ' + fieldName + ';\n}\n'
        return banner