"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""

import numpy as np
import matplotlib.pyplot as plt
import math

grid_size = []
error_values = []

fileName = 'errors.txt'

with open(fileName, 'r') as file:
    for line in file:
        parts = line.split()
        grid_size.append((float(parts[1])))
        error_values.append(float(parts[3]))


fig=plt.figure(figsize=(5,8))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.xlabel('Grid element size',fontsize=12)
plt.ylabel('Error (in %)',fontsize=12)
plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.1)
plt.title('Relative error convergence')
plt.yscale('log')
plt.xscale('log')

plt.plot( \
    grid_size, \
    error_values, \
    color='blue',linestyle='solid',linewidth=2,alpha=1, \
    marker='o', markersize=8, markerfacecolor='none', \
    zorder=1, label='HoPyFOAM'
)

# Theoretical slopes
def order(np, order, offset):
    result = []
    for i in range(len(np)):
        result.append(offset*(np[i])**order)
    return result

L_2 = order(grid_size, 2, error_values[0]*1.3)
L_3 = order(grid_size, 3, error_values[0]*1.3)
L_4 = order(grid_size, 4, error_values[0]*1.3)
plt.plot( grid_size, L_3, '--', color = "orange", linewidth = 1, label='Theoretical slope - 3 order')
plt.plot( grid_size, L_2, '--', color = 'grey', linewidth = 1, label='Theoretical slope - 2 order')
#plt.plot( grid_size, L_4, '--', color = 'red',  linewidth = 1,  label='Theoretical slope - 4 order')

plt.legend()

figure = plt.gcf()
figure.savefig("relErrorConvergence.png", dpi=100)
plt.close(fig)
