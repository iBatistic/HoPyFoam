# HoPyFOAM  
### FOAM like Python code for High-order Finite Volume Method

__HoPyFOAM__ is an OpenFOAM-like Python package written for a high-order finite volume method.  
The code is intended to solve solid mechanics problems on 2D and 3D unstructured meshes.  
It is based on [numpy](https://numpy.org/) and [petsc4py](https://petsc.org/release/petsc4py/) packages while the coding style and code structure 
is similar to one from the [OpenFOAM](https://www.openfoam.com/) library.

### Authors and Contributors
- Ivan Batistić; [ibatistic@fsb.hr](ibatistic@fsb.hr)
- Philip Cardiff; [philip.cardiff@ucd.ie](philip.cardiff@ucd.ie)

### Installation

1. Clone the directory with `git clone git@github.com:iBatistic/HoPyFoam.git`
2. Required python are listed in `requirements.txt` and can be installed easily with `venv`:  
    ```
    virtualenv venv-HoPyFoam
    source venv-HoPyFoam/bin/activate
    pip install -r requirements.txt
    ```

3. Install some of the OpenFOAM.COM distribution using the instructions on this [link](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian). OpenFOAM is required for mesh generation and post-processing functionalities.

4. To run cases, first source OpenFOAM shell session and then activate Python environment:
    ```
    openfoam2312
    source venv-HoPyFoam/bin/activate
    ```
    All tutorial cases have a corresponding `./Allrun` script to run them.   

### License

This toolkit is released under the GNU General Public License (version 3). 
More details can be found in the [LICENSE](./LICENSE.txt) file.

### Disclaimer
This offering is not approved or endorsed by OpenCFD Limited, 
producer and distributor of the OpenFOAM software via (www.openfoam.com)[https://www.openfoam.com/}, 
and owner of the OPENFOAM® and OpenCFD® trade marks.

### Tutorials

- __`laplacianFoam1D`__

    - __`heatConduction.py`__   
        Python script for 1D heat conduction with linear temperature profile.

    - __`heatConductionMMS.py`__  
        Python script for 1D heat conduction with variable source term obtained using MMS.

- __`laplacianFoam`__

    - __`squareBlock2D `__   
        Laplace equation solved on 2D square domain ($1 x 1$ m), left and right patches have values of  $0$ and $1$, top and bottom patches are zero gradient.

 - __`elasticSolidFoam`__

    - __`cantilever2D`__  
        Rectangular beam $50 x 2$ m with a Young’s modulus of $30000$ Pa and a Poisson’s ration of $0.3$. The beam is fixed on the left boundary,
        and a uniform distributed traction of $(0,-2,0)$ Pa is applied to the right boundary; the top and bottom boundaries are traction-free.
        Example is taken from: _Demirdžić, I. "A fourth-order finite volume method for structural analysis." Applied Mathematical Modelling, 2016._
        
    - __`squareBlock2D`__   
        2D square domain ($1 x 1$ m), left patch fixed, top and bottom are traction-free. The right patch have prescribed traction in $x$ direction. Poisson's value is $0$ resulting in linear distribution of displacement field.
    
    - __`squareBlock2DMMS`__  
        2D square domain ($1 x 1$ m), all patches have zero displacement. Body force calculated according to expected solution.  Example is taken from: 
        _Aycock, Kenneth I., Nuno Rebelo, and Brent A. Craven. "Method of manufactured solutions code verification of elastostatic solid mechanics problems in a commercial finite element solver."  Computers & Structures, 2020._
    
        
### Contact, support, and contribution information
To contact the authors about __HoPyFOAM__, please use the issue tracker of the GitHub project. Bug reports and contributions to new features are welcome.
