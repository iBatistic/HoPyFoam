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

1. Clone the directory with `git clone git@github.com:iBatistic/HoPyFVM.git`
2. Required python are listed in `requirements.txt` and can be installed easily with `venv`:  
    ```
    virtualenv --no-site-packages venv-HoPyFoam
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

- __`1D_LaplacianFoam`__

    - __`1D_heatConduction.py`__   
        Python script for 1D heat conduction with linear temperature profile.

    - __`1D_heatConduction_MMS.py` __
        Python script for 1D heat conduction with variable source term and analytical solution obtained using MMS.

- __`LaplacianFoam`__

    - __`example_1`__
        Laplace equation for 2D square domain (1 x 1 m), left and right patches have value of 1, top and bottom patches are zero gradient.

    - __`example_2 `__
        Laplace equation for 2D square domain (1 x 1 m)  with analytical solution from "_I do like CFD, VOL.1, Katate Masatsuka, edition II_"
        page 222, solution c.
        - `example_2/tet`  tetrahedral mesh, coarse and fine mesh available
        - `example_2/hex`  hexahedral discretisation using `blockMesh`
        
        
        

