"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
Description
    Laplace's equation solver
"""
__author__ = 'Ivan Batistić & Philip Cardiff'
__email__  = 'ibatistic@fsb.hr, philip.cardiff@ucd.ie'

import time as timeModule
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../.")
from src.finiteVolume.fvMesh import fvMesh
from src.foam.argList import arg_parser

# Execution start time, used to measured elapsed clock time
exec_start_time = timeModule.perf_counter()

# Get command line arguments
args = arg_parser().parse_args()

print(args)


mesh = fvMesh()



print(f'Execution time = {timeModule.perf_counter() - exec_start_time:.2f} s')
print('End')