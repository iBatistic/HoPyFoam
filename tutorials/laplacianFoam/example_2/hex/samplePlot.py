"""
 _____     _____     _____ _____ _____ _____    |  High-order Python FOAM
|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10
|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0
|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3
                |___|
"""

import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(5,8))

fileName = 'postProcessing/sample/1.0/line_T.xy'

x_values = []
T_values = []

x_values_analytical  = [i * 0.02 for i in range(0, 51)]
T_values_analytical = []

with open(fileName, 'r') as file:
    for line in file:
        parts = line.split()
        x_values.append(float(parts[0]))
        T_values.append(float(parts[1]))

for i,x in enumerate(x_values_analytical ):
    #T_values_analytical.append(x*x - 0.5*0.5)
    #T_values_analytical.append(2.0*0.5 / (pow((1+x),2)+pow(0.5,2)))
    T_values_analytical.append(np.sin(5*0.5) * np.exp(5*x))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.xlabel('$x$-axis',fontsize=12)
plt.ylabel('$y$-axis',fontsize=12)
plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.1)
plt.xlim(0, 1)
plt.title('Results at line y=0.5 m')

plt.plot(x_values_analytical , T_values_analytical, color='red',linestyle='solid',linewidth=2,alpha=0.7,marker='',zorder=2, label='analytical')
plt.plot(x_values, T_values, color='blue',linestyle='solid',linewidth=2,alpha=1,marker='o', markersize=8, markerfacecolor='none', zorder=1, label='results')

plt.legend(shadow=False,fontsize=12)
plt.tight_layout()

figure = plt.gcf()
#plt.show()
figure.savefig("samplePlot_midLine.png", dpi=100)
plt.close(fig)

