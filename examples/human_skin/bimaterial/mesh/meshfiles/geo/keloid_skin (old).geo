
he_default = 1.0;
he_pads = he_default * 0.25;
he_keloid = he_default * 0.5;

// INDEPENDENT VARIABLES

L = 100;
H = 50;

l_pad = 8.0;
h_pad = 8.0;

dx_pad = 32.0; // initial pad separation distance


// CONSTANTS
L0 = 100;
H0 = 40;


// DEPENDENT VARIABLES

// bounding box
dH = H - H0;
dL = L - L0;
x0 = 0.0 - dL*0.5;
y0 = 0.0 - dH*0.5;
x1 = L0 + dL*0.5;
y1 = H0 + dH*0.5;

// pad one
x0_pad_one = L0*0.5 - dx_pad*0.5 - l_pad;
y0_pad_one = H0*0.5 - h_pad*0.5;
x1_pad_one = L0*0.5 - dx_pad*0.5;
y1_pad_one = H0*0.5 + h_pad*0.5;

// pad two
x0_pad_two = L0*0.5 + dx_pad*0.5;
y0_pad_two = H0*0.5 - h_pad*0.5;
x1_pad_two = L0*0.5 + dx_pad*0.5 + l_pad;
y1_pad_two = H0*0.5 + h_pad*0.5;


// outline of keloid
Point(1) = {39.3, 15.4, 0, he_keloid};
Point(2) = {40.8, 15.6, 0, he_keloid};
Point(3) = {42, 15.7, 0, he_keloid};
Point(4) = {43.8, 15.8, 0, he_keloid};
Point(5) = {45.2, 15.85, 0, he_keloid};
Point(6) = {48.15, 15.9, 0, he_keloid};
Point(7) = {51.55, 15.85, 0, he_keloid};
Point(8) = {54.75, 15.7, 0, he_keloid};
Point(9) = {58, 15.4, 0, he_keloid};
Point(10) = {59.05, 15.2, 0, he_keloid};
Point(11) = {60.5, 14.45, 0, he_keloid};
Point(12) = {62.4, 13.45, 0, he_keloid};
Point(13) = {65.1, 12.05, 0, he_keloid};
Point(14) = {67, 11.55, 0, he_keloid};
Point(15) = {67.9, 11.4, 0, he_keloid};
Point(16) = {69.1, 11.45, 0, he_keloid};
Point(17) = {70.15, 11.65, 0, he_keloid};
Point(18) = {71.55, 12, 0, he_keloid};
Point(19) = {72.75, 12.35, 0, he_keloid};
Point(20) = {74.05, 13.2, 0, he_keloid};
Point(21) = {75, 14.15, 0, he_keloid};
Point(22) = {75.85, 15.4, 0, he_keloid};
Point(23) = {76.35, 16.7, 0, he_keloid};
Point(24) = {76.6, 18.05, 0, he_keloid};
Point(25) = {76.7, 19.3, 0, he_keloid};
Point(26) = {76.5, 21, 0, he_keloid};
Point(27) = {76.15, 22.55, 0, he_keloid};
Point(28) = {75.7, 23.8, 0, he_keloid};
Point(29) = {74.85, 25.15, 0, he_keloid};
Point(30) = {74, 25.95, 0, he_keloid};
Point(31) = {73.35, 26.65, 0, he_keloid};
Point(32) = {71.85, 27.2, 0, he_keloid};
Point(33) = {70.55, 27.3, 0, he_keloid};
Point(34) = {69.8, 27.3, 0, he_keloid};
Point(35) = {68.65, 27.25, 0, he_keloid};
Point(36) = {67.9, 27.15, 0, he_keloid};
Point(37) = {66.1, 26.8, 0, he_keloid};
Point(38) = {64.7, 26.6, 0, he_keloid};
Point(39) = {62.45, 26.2, 0, he_keloid};
Point(40) = {60.05, 25.6, 0, he_keloid};
Point(41) = {58.4, 25.2, 0, he_keloid};
Point(42) = {57.5, 25.1, 0, he_keloid};
Point(43) = {55.45, 25, 0, he_keloid};
Point(44) = {53.55, 25, 0, he_keloid};
Point(45) = {50.6, 25.05, 0, he_keloid};
Point(46) = {46.75, 25.05, 0, he_keloid};
Point(47) = {43.8, 25.05, 0, he_keloid};
// Point(48) = {42.8, 25.05, 0, he_keloid};
Point(49) = {40.75, 25.0, 0, he_keloid};
Point(50) = {38.75, 25.50, 0, he_keloid};
Point(51) = {36.1, 26.35, 0, he_keloid};
Point(52) = {34.35, 27.25, 0, he_keloid};
Point(53) = {33.5, 27.55, 0, he_keloid};
Point(54) = {32.6, 27.7, 0, he_keloid};
Point(55) = {31.7, 27.8, 0, he_keloid};
Point(56) = {30.8, 27.75, 0, he_keloid};
Point(57) = {29.9, 27.7, 0, he_keloid};
Point(58) = {29, 27.55, 0, he_keloid};
Point(59) = {28.15, 27.3, 0, he_keloid};
Point(60) = {27.3, 27.05, 0, he_keloid};
Point(61) = {26.05, 26.55, 0, he_keloid};
Point(62) = {24.8, 25.7, 0, he_keloid};
Point(63) = {24.05, 25.05, 0, he_keloid};
Point(64) = {23.6, 24.25, 0, he_keloid};
Point(65) = {23.25, 23.35, 0, he_keloid};
Point(66) = {23.05, 22.45, 0, he_keloid};
Point(67) = {22.95, 21.45, 0, he_keloid};
Point(68) = {22.85, 20.45, 0, he_keloid};
Point(69) = {22.85, 19.55, 0, he_keloid};
Point(70) = {22.85, 18.6, 0, he_keloid};
Point(71) = {22.95, 17.65, 0, he_keloid};
Point(72) = {23.15, 16.7, 0, he_keloid};
Point(73) = {23.55, 15.8, 0, he_keloid};
Point(74) = {24.15, 15.05, 0, he_keloid};
Point(75) = {24.9, 14.45, 0, he_keloid};
Point(76) = {25.7, 13.8, 0, he_keloid};
Point(77) = {26.4, 13.2, 0, he_keloid};
Point(78) = {27.15, 12.6, 0, he_keloid};
Point(79) = {28, 12.25, 0, he_keloid};
Point(80) = {28.85, 12.05, 0, he_keloid};
Point(81) = {29.75, 11.85, 0, he_keloid};
Point(82) = {30.7, 11.8, 0, he_keloid};
Point(83) = {31.65, 11.85, 0, he_keloid};
Point(84) = {32.55, 11.9, 0, he_keloid};
Point(85) = {33.45, 12.1, 0, he_keloid};
Point(86) = {34.3, 12.3, 0, he_keloid};
Point(87) = {35.2, 12.75, 0, he_keloid};
Point(88) = {36, 13.3, 0, he_keloid};
Point(89) = {36.85, 13.9, 0, he_keloid};
Point(90) = {37.55, 14.55, 0, he_keloid};
Point(91) = {38.35, 15.1, 0, he_keloid};


// bounding box
Point(92) = {x0, y0, 0, he_default};
Point(93) = {x1, y0, 0, he_default};
Point(94) = {x1, y1, 0, he_default};
Point(95) = {x0, y1, 0, he_default};

// pad one
Point(96) = {x0_pad_one, y0_pad_one, 0, he_pads};
Point(97) = {x1_pad_one, y0_pad_one, 0, he_pads};
Point(98) = {x1_pad_one, y1_pad_one, 0, he_pads};
Point(99) = {x0_pad_one, y1_pad_one, 0, he_pads};

// pad two
Point(100)  = {x0_pad_two, y0_pad_two, 0, he_pads};
Point(101) = {x1_pad_two, y0_pad_two, 0, he_pads};
Point(102) = {x1_pad_two, y1_pad_two, 0, he_pads};
Point(103) = {x0_pad_two, y1_pad_two, 0, he_pads};


// bounding box lines
Spline(1) = {92, 93};
Spline(2) = {93, 94};
Spline(3) = {94, 95};
Spline(4) = {95, 92};

Spline(5) = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

// pad one
Line(6) = {96, 97};
Line(7) = {97, 98};
Line(8) = {98, 99};
Line(9) = {99, 96};

// pad two
Line(10) = {100, 101};
Line(11) = {101, 102};
Line(12) = {102, 103};
Line(13) = {103, 100};

// Healthy skin subdomain
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5};
Plane Surface(1) = {1, 2};

// Keloid subdomain
Line Loop(3) = {10, 11, 12, 13};
Line Loop(4) = {7, 8, 9, 6};
Plane Surface(2) = {2, 3, 4};

Physical Line("pad_one", 5) = {6, 7, 8, 9};
Physical Line("pad_two", 6) = {10, 11, 12, 13};

Physical Line("bottom", 1) = {6, 7, 8, 9};
Physical Line( "right", 2) = {10, 11, 12, 13};
Physical Line(   "top", 3) = {6, 7, 8, 9};
Physical Line(  "left", 4) = {10, 11, 12, 13};

Physical Surface("keloid_skin", 1) = {2};
Physical Surface("healthy_skin", 2) = {1};
