// Mesh spacing
dx = 0.999;
//dx = 0.4;
//dx = 0.25;
//dx = 0.166;

// Beam length
L = 50;

// Beam thickness
t = 2;

// Depth (out of plane)
d = 1;

// Points
Point(1) = {0, -t/2, 0, dx};
Point(2) = {L, -t/2, 0, dx};
Point(3) = {L, t/2, 0, dx};
Point(4) = {0, t/2, 0, dx};

// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Surface
Line Loop(6) = {4, 1, 2, 3};
Plane Surface(6) = {6};

// Force mapped meshing (triangles)
Transfinite Surface {6};
// Uncomment this for unstructured mesh
//Mesh.Algorithm = 6;

// Optional: combine triangles into quadrilaterals
//Recombine Surface {6};

// Create volume by extrusion
Physical Volume("internal") = {1};
Extrude {0, 0, d} {
 Surface{6};
 Layers{1};
 Recombine;
}

// Boundary patches
Physical Surface("front") = {28};
Physical Surface("back") = {6};
Physical Surface("top") = {27};
Physical Surface("left") = {15};
Physical Surface("bottom") = {19};
Physical Surface("right") = {23};
