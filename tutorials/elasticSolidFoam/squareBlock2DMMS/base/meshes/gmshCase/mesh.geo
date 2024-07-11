// Mesh spacing
dx = 0.035;
//dx = 0.45;
//dx = 0.275;
//dx = 0.178;

// Domain length
L = 1;

// Domain thickness
t = 1;

// Depth (out of plane)
d = 1;

// Points
Point(1) = {0, 0, 0, dx};
Point(2) = {L, 0, 0, dx};
Point(3) = {L, t, 0, dx};
Point(4) = {0, t, 0, dx};

// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Surface
Line Loop(6) = {4, 1, 2, 3};
Plane Surface(6) = {6};

// Force mapped meshing (triangles)
//Transfinite Surface {6};
Mesh.Algorithm = 6;

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
