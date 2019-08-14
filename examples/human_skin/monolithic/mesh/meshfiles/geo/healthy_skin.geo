
//he_default = 1.25;
//he_default = 1.00;
//he_default = 0.75;
//he_default = 0.50;
he_default = 0.25;

he_pads = he_default * 1;


// INDEPENDENT VARIABLES

L0 = 100;
H0 = 40;

L = 100;
H = 50;

l_pad = 8.0; // pad length
h_pad = 8.0; // pad width

dL_pad = 32.0; // initial pad separation distance



// DEPENDENT VARIABLES

// bounding box
dH = H - H0;
dL = L - L0;
x0 = 0.0 - dL*0.5;
y0 = 0.0 - dH*0.5;
x1 = L0 + dL*0.5;
y1 = H0 + dH*0.5;

// pad one
x0_pad_one = L0*0.5 - dL_pad*0.5 - l_pad;
y0_pad_one = H0*0.5 - h_pad*0.5;
x1_pad_one = L0*0.5 - dL_pad*0.5;
y1_pad_one = H0*0.5 + h_pad*0.5;

// pad two
x0_pad_two = L0*0.5 + dL_pad*0.5;
y0_pad_two = H0*0.5 - h_pad*0.5;
x1_pad_two = L0*0.5 + dL_pad*0.5 + l_pad;
y1_pad_two = H0*0.5 + h_pad*0.5;

// bounding box
Point(1) = {x0, y0, 0, he_default};
Point(2) = {x1, y0, 0, he_default};
Point(3) = {x1, y1, 0, he_default};
Point(4) = {x0, y1, 0, he_default};

// pad one
Point(5) = {x0_pad_one, y0_pad_one, 0, he_pads};
Point(6) = {x1_pad_one, y0_pad_one, 0, he_pads};
Point(7) = {x1_pad_one, y1_pad_one, 0, he_pads};
Point(8) = {x0_pad_one, y1_pad_one, 0, he_pads};

// pad two
Point(9)  = {x0_pad_two, y0_pad_two, 0, he_pads};
Point(10) = {x1_pad_two, y0_pad_two, 0, he_pads};
Point(11) = {x1_pad_two, y1_pad_two, 0, he_pads};
Point(12) = {x0_pad_two, y1_pad_two, 0, he_pads};

// bounding box lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// pad one
Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 5};

// pad two
Line(10) = { 9, 10};
Line(11) = {10, 11};
Line(12) = {11, 12};
Line(13) = {12, 9};

Physical Line("boundary_bottom", 1) = {1};
Physical Line("boundary_right", 2) = {2};
Physical Line("boundary_top", 3) = {3};
Physical Line("boundary_left", 4) = {4};

Physical Line("moving_pad", 5) = {6, 7, 8, 9};
Physical Line("fixed_pad", 6) = {10, 11, 12, 13};

Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {6, 7, 8, 9};
Line Loop(4) = {10, 11, 12, 13};

Plane Surface(1) = {1, 2, 4};
Physical Surface("healthy_skin", 1) = {1};
