# __1D steady-state heat conduction (Laplace equation)__

Laplace equation for heat conduction:

$$
-k\nabla^2 T = 0
$$

After integration:

$$
-\int_{\Omega} k \ \nabla \cdot \nabla T = 0
$$

After applying div theorem:

$$
-\oint_{\Gamma} k \ \nabla T \cdot \textbf{n} \ \text{d} \Gamma = 0
$$

$$

$$

### __FVM discretisation__

$$
-\sum_{f=1}^{f=N_f} \int_{\Gamma_f} k \ \nabla T \cdot \textbf{n}_f \ \text{d} \Gamma_f = 0
$$

After numerical quadrature:

$$
- \sum_{f=1}^{f=N_f} k_f \underbrace{\left[ \sum_{g=1}^{g=N_g}  \alpha_g \nabla T (\textbf{x}_{f,g}) \cdot \textbf{n}_f \right]}_{\displaystyle\int_{\Gamma_f} \nabla T \cdot \textbf{n}_f} \Gamma_f = 0
$$

For 1D above equation transform to ($N_g=1, \alpha_g=1$):

$$
- \sum_{f=1}^{f=N_f} k_f \left[    \nabla T (\textbf{x}_f) \cdot \textbf{n}_f \right] \Gamma_f = 0\tag{A3}
$$

__Interpolation scheme for $\nabla T$__

Temperature gradient at face is obtained multiplying each neighboring node in stencil with corresponding weight:

$$
\displaystyle\frac{\partial T}{\partial x}(\tilde{x}) = \sum_{n=1}^{n=N_n}c_{x,n}(\mathbf{\tilde{x}}){T}_n\tag{A4}
$$

- $\mathbf{\tilde{x}}$ is the field point location
- $T_N$ is temperature at the neighbor cell centre
- $\textbf{x}_n$ is location of cell centre

For 1D problem, scalar product of temperature gradient and outward pointing normal vector reduces to:

$$
\nabla T \cdot \textbf{n}_{f} = \frac{\partial T}{\partial x} n_{f,x}
$$

Using eq. (A4) above equation is transformed to:

$$
\nabla T \cdot \textbf{n}_{f} = \left[\sum_{n=1}^{n=N_n}c_{x,n}(\mathbf{\tilde{x}}){T}_n\right]n_{f,x}\tag{A5}
$$

Finally, discretised Laplace equation in 1D is obtained substituting (A5) to (A3)

$$
\displaystyle- \sum_{f=1}^{f=N_f} k_f   \left[\sum_{n=1}^{n=N_n}c_{x,n}(\mathbf{\tilde{x}}){T}_n\right]n_{f,x}  \Gamma_f = 0\tag{A6}
$$

__Weight function__

Exponential kernel (radially symmetric exponential function):
$$
w(\textbf{x}_n, \textbf{x}, k) = \frac{e^{-\left( \frac{d}{d_m}\right)^2k^2}-e^{-k^2}}{-e^{-k^2}} \text{\quad or like this\quad } w = \frac{e^{-\left( \frac{d}{c}\right)^2}-e^{-\left(\frac{d_m}{c}\right)^2}}{-e^{-\left(\frac{d_m}{c}\right)^2}}
$$
$d = ||\textbf{x}_n - \textbf{x}||$ 

$d_m =2 \ \text{max}(\||\textbf{x}_f -\textbf{x}_n||) = 2 \ r_s$

$k=6$ is shape parameter

$k = d_m /c$ where $d_m$ is smoothing length $d_m = r_s$ and $c=d_m/s_x$ where $s_x$ is shape parameter of the kernel (some constant)



__Local Regression Estimators__

Truncated Taylor expansion using $N_p$ terms:

$$
\tilde{T}(x) = T({\tilde{x}}) + \frac{\partial T}{\partial x}({\tilde{x}})(x-\tilde{x}) ~~...
$$

$$
\mathbf{q}^T (\mathbf{x}-\mathbf{\tilde{x}}) = [1, ~~(x-\tilde{x}), ~...] \\
\mathbf{\tilde{a}}^T (\mathbf{\tilde{x}}) = [T({\tilde{x}}),  ~~\frac{\partial T}{\partial x}({\tilde{x}}), ~...]
$$

$$
\mathcal{R} = \frac{1}{2} \sum_{n=1}^{N=N_n}w(\mathbf{x}_n-\mathbf{\tilde{x}})[\tilde{T}(\mathbf{x}_n)-T_n]^2 \\
 = \frac{1}{2} \sum_{n=1}^{N=N_n}w(\mathbf{x}_n-\mathbf{\tilde{x}})[\mathbf{q}^T(\mathbf{x}_n-\mathbf{\tilde{x}})\mathbf{\tilde{a}}(\mathbf{\tilde{x}})-T_n]^2
$$

$$
\frac{\partial \mathcal{R}}{\partial a} = \sum_{n=1}^{N=N_n}w(\mathbf{x}_n-\mathbf{\tilde{x}})[\mathbf{q}^T(\mathbf{x}-\mathbf{\tilde{x}})\mathbf{\tilde{a}}(\mathbf{\tilde{x}})-T_n] = 0 \\
w(\mathbf{x}_n-\mathbf{\tilde{x}}) = \text{n-th column of } \mathbf{W} \\
\mathbf{q}(\mathbf{x}_n-\mathbf{\tilde{x}}) = \text{n-th column of } \mathbf{Q} \\
0 = \mathbf{W}[\mathbf{Q}^T\mathbf{\tilde{a}} - \mathbf{T}_n] \\
\mathbf{W}\mathbf{Q}^T\mathbf{\tilde{a}} =\mathbf{W}\mathbf{T}_n \quad /\mathbf{Q}\\
\mathbf{Q} \mathbf{W} \mathbf{Q}^T \mathbf{\tilde{a}} = \mathbf{Q} \mathbf{W} \mathbf{T}_n \\
\mathbf{\tilde{M}} = \mathbf{Q} \mathbf{W} \mathbf{Q}^T\\
\mathbf{\tilde{A}} = \mathbf{\tilde{M}}^{-1} \mathbf{Q} \mathbf{W} \\
\mathbf{\tilde{a}} = \mathbf{\tilde{A}}\mathbf{T}_n
$$

$$
\begin{bmatrix}
T(\mathbf{\tilde{x}})\\
\displaystyle\frac{\partial T}{\partial x}(\mathbf{\tilde{x}})\\
\displaystyle\frac{\partial T}{\partial y}(\mathbf{\tilde{x}}) \\
\vdots \\
\displaystyle\frac{\partial^2T}{\partial y^2}(\mathbf{\tilde{x}}) 
\end{bmatrix}
=
\mathbf{\tilde{A}}\mathbf{T}_n
$$



On boundary:
$$
\tilde{T}(x) = T({\tilde{x}}) + \frac{\partial T}{\partial x}({\tilde{x}})\cancel{(x-\tilde{x})}^0 + ~ ... = T({\tilde{x}})
$$

$$
\mathbf{q}^T (\mathbf{x}-\mathbf{\tilde{x}}) = [1, ~~0, ...] \\
\mathbf{\tilde{a}}^T (\mathbf{\tilde{x}}) = [T({\tilde{x}}),  ~~\frac{\partial T}{\partial x}({\tilde{x}}), ...]
$$

For $N_p = 2$ (same is for $N_p>2$ )
$$
w(\mathbf{x}_n-\mathbf{\tilde{x}}) = \text{n-th column of } \mathbf{W} \\
\mathbf{q}(\mathbf{x}_n-\mathbf{\tilde{x}}) = \text{n-th column of } \mathbf{Q} \\
W = \begin{bmatrix} w(\mathbf{x}_n-\mathbf{\tilde{x}}) &  0 \\
0 & w(\mathbf{x}_n-\mathbf{\tilde{x}})\end{bmatrix}
=
\begin{bmatrix} w(\mathbf{x}_n-\mathbf{\tilde{x}}) &  0 \\
0 & w(0)\end{bmatrix}
=
\begin{bmatrix} w(\mathbf{x}_n-\mathbf{\tilde{x}}) &  0 \\
0 & 1\end{bmatrix}\\
\\
Q = \begin{bmatrix} 1 &  1 \\
\mathbf{x}_n-\mathbf{\tilde{x}} & 0\end{bmatrix}
$$

$$
\begin{bmatrix}
T(\mathbf{\tilde{x}})\\
\displaystyle\frac{\partial T}{\partial x}(\mathbf{\tilde{x}})
\end{bmatrix}
=
(\mathbf{Q} \mathbf{W} \mathbf{Q}^T)^{-1}\mathbf{Q}\mathbf{W}
\begin{bmatrix}
T_0\\
T_b
\end{bmatrix}
=
\begin{bmatrix}
A_{11} & A_{12}\\
A_{21} & A_{22}
\end{bmatrix}
\begin{bmatrix}
T_0\\
T_b
\end{bmatrix}
$$


$$
\displaystyle\frac{\partial T}{\partial x}(\mathbf{\tilde{x}}) = \underbrace{T_0 * A_{21}}_{\text{diag coeff}} + \underbrace{T_b * A_{22}}_{\text{source vector}}
$$




__Solution procedure__


1. step: Loop over interior cells

$$
\sum_{f=w,e}k_f {n}_{fx}  \Gamma_f \displaystyle\sum_{n=1}^{n=N_n} \textbf{c}_{{x}, n}
$$

2. step: Loop over boundary faces
3. Solve $\mathbf{A}\mathbf{x}=\mathbf{b}$ system of equations

---

# __Example 1__

1D rod with constant cross-section area with prescribed temperature at end points. Without volume source term.

__INPUT:__
Cross-section area $\Gamma_f = 10 $
Diffusion coefficient $k_f = 10  $
Overall rod length $L=10$
Number of CVs = 10
$\delta x = 1$
$T_A = 0$
$T_B = 10$
Outward pointing normal at east face is $n_e = 1$
Outward pointing normal at west face is $n_e = -1$

# __Example 2__

1D rod with constant cross-section area with prescribed zero temperature at end points.

Volume source term is calculated using **MMS (Method of Manufactured Solutions)**

Expected solution:
$$
T = \sin\left(2\pi\frac{x^2}{100}\right)
$$
The source term is then found by substituting the manufactured expression for T into the governing equation:
$$
\frac{\text{d}}{\text{d} x} \frac{\text{d}T}{\text{d}x} = -\frac{1}{625} \pi \left(\pi x^2\sin\left(\frac{\pi x^2}{50}\right) - 25\cos \left(\frac{\pi x^2}{50}\right) \right)
$$
Analytical integration for each cell:
$$
\int_a^b -\frac{1}{625} \pi \left(\pi x^2\sin\left(\frac{\pi x^2}{50}\right) - 25\cos \left(\frac{\pi x^2}{50}\right) \right) \text{d} x = \frac{1}{25}\left( \pi b \cos \left(\frac{\pi b^2}{50} \right) - \pi a \cos \left(\frac{\pi a^2}{50}\right) \right)
$$


__INPUT:__
Cross-section area $\Gamma_f = 10 $
Diffusion coefficient $k_f = 10$
Overall rod length $L=10$
Number of CVs = 10
$\delta x = 1$
$T_A = 0$
$T_B = 10$
Outward pointing normal at east face is $n_e = 1$
Outward pointing normal at west face is $n_e = -1$  

