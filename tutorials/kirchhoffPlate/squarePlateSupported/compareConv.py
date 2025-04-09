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
error_values_N1 = []
error_values_N2 = []
error_values_N3 = []

fileName = 'errors_N1.txt'
fileName2 = 'errors_N2.txt'
fileName3 = 'errors_N3.txt'

with open(fileName, 'r') as file:
    for line in file:
        parts = line.split()
        grid_size.append(float(parts[1]))
        error_values_N1.append(float(parts[2]))
        
with open(fileName2, 'r') as file:
    for line in file:
        parts = line.split()
        error_values_N2.append(float(parts[2]))
        
with open(fileName3, 'r') as file:
    for line in file:
        parts = line.split()
        error_values_N3.append(float(parts[2]))
        
fig=plt.figure(figsize=(5,8))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.xlabel('Grid size',fontsize=12)
plt.ylabel('Error',fontsize=12)
plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.1)
plt.title('Absolute error convergence')
plt.yscale('log')
plt.xscale('log')

plt.plot( \
    grid_size, \
    error_values_N1, \
    color='blue',linestyle='solid',linewidth=2.2,alpha=1, \
    marker='o', markersize=8, markerfacecolor='none', \
    zorder=1, label='HoPyFOAM - 2 order'
)

plt.plot( \
    grid_size, \
    error_values_N2, \
    color='green',linestyle='solid',linewidth=2.2,alpha=1, \
    marker='o', markersize=8, markerfacecolor='none', \
    zorder=1, label='HoPyFOAM - 3 order'
)

plt.plot( \
    grid_size, \
    error_values_N3, \
    color='red',linestyle='solid',linewidth=2.2,alpha=1, \
    marker='o', markersize=8, markerfacecolor='none', \
    zorder=1, label='HoPyFOAM - 4 order'
)



# Theoretical slopes
def order(np, order, offset):
    result = []
    for i in range(len(np)):
        result.append(offset*(np[i])**order)
    return result

L_2 = order(grid_size, 2, error_values_N1[0]*1.1)
L_2_t = order(grid_size, 2, error_values_N1[0]*0.172)
L_3 = order(grid_size, 3, error_values_N3[0]*1.51)
L_4 = order(grid_size, 4, error_values_N3[0]*1.51)
plt.plot( grid_size, L_2, '--', color = "black", linewidth = 1.5, label='Theoretical slope - 2 order')
plt.plot( grid_size, L_2_t, '--', color = "black", linewidth = 1.5, label='')
plt.plot( grid_size, L_3, '--', color = 'orange', linewidth = 1.5, label='Theoretical slope - 3 order')
plt.plot( grid_size, L_4, '--', color = 'green',  linewidth = 1.5, label='Theoretical slope - 4 order')

plt.legend()

figure = plt.gcf()
figure.savefig("absErrorConvergenceComparison.png", dpi=150)
plt.close(fig)
