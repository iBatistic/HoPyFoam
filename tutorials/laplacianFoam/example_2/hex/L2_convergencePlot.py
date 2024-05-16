import numpy as np
import matplotlib.pyplot as plt
import random

# Plotting the plot
fig=plt.figure(figsize=(5,8))

# Values
mesh_size = [25, 100, 400, 1600]
# T = X^2 + Y^2
#L_2_p1 = [3.9987e-03, 1.1829e-03,  3.1056e-04, 7.8799e-05]
#L_2_p1 = [3.9786e-03, 1.1749e-03,  3.0867e-04, -] # ovo je sa 4 susjeda, a iznad je sa 2!
#L_2_p2 = [1e-10, 1e-10,  1.0000e-10, 1.0000e-10]

# T = sin(y) * e^x
L_2_p1 = [2.1238e-03, 6.2617e-04,  1.6489e-04, 4.1907e-05]
L_2_p2 = [3.5703e-04, 4.8823e-05,  9.0460e-06, 2.0094e-06]
#L_2_p3 = [1.0936e-04, 2.1644e-05,  3.8300e-06, 5.5835e-07]
L_2_p3 = [5.6958e-05, 3.9375e-06,  3.0030e-07, 2.0880e-08]

def order(np, order, offset):
    result = []
    for i in range(len(np)):
        result.append(offset*(1/np[i])**order)
    return result

L_2_i1 = order(mesh_size, 1, 0.05)
L_2_i2 = order(mesh_size, 2, 0.100)
L_2_i3 = order(mesh_size, 3, 2)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

plt.yscale('log')
plt.xscale('log')
plt.plot(mesh_size, L_2_p1, color='blue',linestyle='solid', linewidth=3, alpha=0.8,marker='o', markersize=8, markerfacecolor='none', zorder=6, label='second order')
plt.plot(mesh_size, L_2_p2, color='red',linestyle='solid', linewidth=3, alpha=0.8,marker='o', markersize=8, markerfacecolor='none', zorder=6, label='third order')
plt.plot(mesh_size, L_2_p3, color='orange',linestyle='solid', linewidth=3, alpha=0.8,marker='o', markersize=8, markerfacecolor='none', zorder=6, label='fourth order')

plt.plot( mesh_size, L_2_i1, '--', color = "blue", linewidth = 1, label='second order slope')
plt.plot( mesh_size, L_2_i2, '--', color = 'red', linewidth = 1, label='third order slope')
plt.plot( mesh_size, L_2_i3, '--', color = 'orange', linewidth = 1, label='fourth order slope')

plt.title('$L_2$ norm convergence plot (log-log)', fontsize=12)

#plt.xlim(0, 6)
plt.ylim(1e-8, 0.5e-2)
#plt.axis('equal')
#plt.axis('scale')

plt.xlabel('Number of CVs',fontsize=12)
plt.ylabel('$L_2$',fontsize=12)
plt.legend(shadow=False,fontsize=12)
plt.legend(loc='best')
plt.grid(color='grey', linestyle='-', linewidth=0.1)
plt.grid(True, which='major', color='black', linestyle='-', linewidth=0.1)
figure = plt.gcf()
plt.tight_layout()
figure.savefig("L2convergence.png", dpi=300)
plt.close(fig)

