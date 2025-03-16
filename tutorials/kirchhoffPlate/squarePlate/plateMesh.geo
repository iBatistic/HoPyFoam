// Gmsh .geo file to create a 2D heat conduction model with extrusion

// Mesh spacing parameter
dx = 0.7; // Mesh size

// Dimensions
L = 10;  // Length in x-direction
W = 10;  // Width in y-direction
d = 1; // Extrusion depth in z-direction

// Define points
Point(1) = {0, 0, 0, dx};    // Bottom-left corner
Point(2) = {0, L, 0, dx};    // Bottom-right corner
Point(3) = {W, L, 0, dx};    // Top-right corner
Point(4) = {W, 0, 0, dx};    // Top-left corner

// Define lines
Line(1) = {1, 2}; // Bottom edge
Line(2) = {2, 3}; // Right edge
Line(3) = {3, 4}; // Top edge
Line(4) = {4, 1}; // Left edge

// Define surface
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Transfinite meshing
//Transfinite Line {1, 2, 3, 4} = 8 Using Progression 1; // Control mesh density
Transfinite Surface {6}; // Structured meshing, keeps control over triangulation

// Extrude to create volume
Extrude {0, 0, d} {
  Surface{6}; // Extrude surface into 3D
  Layers{1};  // Number of layers in extrusion
  Recombine;
}

// Define physical groups for boundary conditions
Physical Surface("back") = {28};   // Front and back surface
Physical Surface("front") = {6};   // Front and back surface
Physical Surface("top") = {19};    // Top surface
Physical Surface("bottom") = {27}; // Bottom surface
Physical Surface("left") = {15};   // Left surface
Physical Surface("right") = {23};  // Right surface
Physical Volume("internal") = {1}; // Internal volume
