"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Surface field class
"""
__author__ = 'Ivan Batistić'
__email__ = 'ibatistic@fsb.unizg.hr'
__all__ = ['surfaceField']

import os
import shutil
import numpy as np
import warnings

from src.foam.field import field
from src.foam.foamFileParser import *
from src.foam.field import localRegressionEstimator
from src.foam.field.volField.volFieldBoundaryConditions import *
from src.foam.decorators import timed

class surfaceField(field):

    def __init__(self, fieldName):

        super().__init__(fieldName)