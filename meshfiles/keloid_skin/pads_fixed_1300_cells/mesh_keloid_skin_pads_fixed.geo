
// INDEPENDENT VARIABLES

L = 100;
H = 50;
lc = 1.2;
lc2 = 0.6;
H_msr = 11.0;

l_pad = 17.0; // pad length
h_pad = 24.0; // pad width
h_pad_square = 8.0; // pad width
h_pad_min = 7.0; // U pad arm width
l_pad_min = 9.0; // U pad arm length
gap_inter_pad = 1.0; // the gap between the square pad and the U-pad

dx_pad = 36.0; // initial pad separation distance


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

x1_pad_one = x0_pad_one + l_pad;
y1_pad_one = y0_pad_one;

x2_pad_one = x1_pad_one;
y2_pad_one = y1_pad_one + h_pad_min;

x3_pad_one = x2_pad_one - l_pad_min;
y3_pad_one = y2_pad_one;

x4_pad_one = x3_pad_one;
y4_pad_one = y3_pad_one + h_pad_square + 2*gap_inter_pad;

x5_pad_one = x4_pad_one + l_pad_min;
y5_pad_one = y4_pad_one;

x6_pad_one = x5_pad_one;
y6_pad_one = y5_pad_one + h_pad_min;

x7_pad_one = x6_pad_one - l_pad;
y7_pad_one = y6_pad_one;

x0_pad_one_square = L0*0.5 - dx_pad*0.5 - h_pad_square;
y0_pad_one_square = H0*0.5 - h_pad_square*0.5;

x1_pad_one_square = x0_pad_one_square + h_pad_square;
y1_pad_one_square = y0_pad_one_square;

x2_pad_one_square = x1_pad_one_square;
y2_pad_one_square = y1_pad_one_square + h_pad_square;

x3_pad_one_square = x2_pad_one_square - h_pad_square;
y3_pad_one_square = y2_pad_one_square;
//*************************************************************
// pad two
x0_pad_two = L0*0.5 + dx_pad*0.5 + l_pad;
y0_pad_two = H0*0.5 - h_pad*0.5;

x1_pad_two = x0_pad_two - l_pad;
y1_pad_two = y0_pad_two;

x2_pad_two = x1_pad_two;
y2_pad_two = y1_pad_two + h_pad_min;

x3_pad_two = x2_pad_two + l_pad_min;
y3_pad_two = y2_pad_two;

x4_pad_two = x3_pad_two;
y4_pad_two = y3_pad_two + h_pad_square + 2*gap_inter_pad;

x5_pad_two = x4_pad_two - l_pad_min;
y5_pad_two = y4_pad_two;

x6_pad_two = x5_pad_two;
y6_pad_two = y5_pad_two + h_pad_min;

x7_pad_two = x6_pad_two + l_pad;
y7_pad_two = y6_pad_two;

x0_pad_two_square = L0*0.5 + dx_pad*0.5 + h_pad_square;
y0_pad_two_square = H0*0.5 - h_pad_square*0.5;

x1_pad_two_square = x0_pad_two_square - h_pad_square;
y1_pad_two_square = y0_pad_two_square;

x2_pad_two_square = x1_pad_two_square;
y2_pad_two_square = y1_pad_two_square + h_pad_square;

x3_pad_two_square = x2_pad_two_square + h_pad_square;
y3_pad_two_square = y2_pad_two_square;


// outline of keloid
Point(1) = {39.3, 15.4, 0, lc2};
Point(2) = {40.8, 15.6, 0, lc2};
Point(3) = {42, 15.7, 0, lc2};
Point(4) = {43.8, 15.8, 0, lc2};
Point(5) = {45.2, 15.85, 0, lc2};
Point(6) = {48.15, 15.9, 0, lc2};
Point(7) = {51.55, 15.85, 0, lc2};
Point(8) = {54.75, 15.7, 0, lc2};
Point(9) = {58, 15.4, 0, lc2};
Point(10) = {59.05, 15.2, 0, lc2};
Point(11) = {60.5, 14.45, 0, lc2};
Point(12) = {62.4, 13.45, 0, lc2};
Point(13) = {65.1, 12.05, 0, lc2};
Point(14) = {67, 11.55, 0, lc2};
Point(15) = {x1_pad_two, 11.4, 0, lc2};
Point(16) = {69.1, 11.45, 0, lc2};
Point(17) = {70.15, 11.65, 0, lc2};
Point(18) = {71.55, 12, 0, lc2};
Point(19) = {72.75, 12.35, 0, lc2};
Point(20) = {74.05, 13.2, 0, lc2};
Point(21) = {75, 14.15, 0, lc2};
Point(22) = {75.85, 15.4, 0, lc2};
Point(23) = {76.35, 16.7, 0, lc2};
Point(24) = {76.6, 18.05, 0, lc2};
Point(25) = {76.7, 19.3, 0, lc2};
Point(26) = {76.5, 21, 0, lc2};
Point(27) = {76.15, 22.55, 0, lc2};
Point(28) = {75.7, 23.8, 0, lc2};
Point(29) = {74.85, 25.15, 0, lc2};
Point(30) = {74, 25.95, 0, lc2};
Point(31) = {73.35, 26.65, 0, lc2};
Point(32) = {71.85, 27.2, 0, lc2};
Point(33) = {70.55, 27.3, 0, lc2};
Point(34) = {69.8, 27.3, 0, lc2};
Point(35) = {68.65, 27.25, 0, lc2};
Point(36) = {x1_pad_two, 27.15, 0, lc2};
Point(37) = {66.1, 26.8, 0, lc2};
Point(38) = {64.7, 26.6, 0, lc2};
Point(39) = {62.45, 26.2, 0, lc2};
Point(40) = {60.05, 25.6, 0, lc2};
Point(41) = {58.4, 25.2, 0, lc2};
Point(42) = {57.5, 25.1, 0, lc2};
Point(43) = {55.45, 25, 0, lc2};
Point(44) = {53.55, 25, 0, lc2};
Point(45) = {50.6, 25.05, 0, lc2};
Point(46) = {46.75, 25.05, 0, lc2};
Point(47) = {43.8, 25.05, 0, lc2};
// Point(48) = {42.8, 25.05, 0, lc2};
Point(49) = {40.75, 25.0, 0, lc2};
Point(50) = {38.75, 25.50, 0, lc2};
Point(51) = {36.1, 26.35, 0, lc2};
Point(52) = {34.35, 27.25, 0, lc2};
Point(53) = {33.5, 27.55, 0, lc2};
Point(54) = {32.6, 27.7, 0, lc2};
Point(55) = {x1_pad_one, 27.8, 0, lc2};
Point(56) = {30.8, 27.75, 0, lc2};
Point(57) = {29.9, 27.7, 0, lc2};
Point(58) = {29, 27.55, 0, lc2};
Point(59) = {28.15, 27.3, 0, lc2};
Point(60) = {27.3, 27.05, 0, lc2};
Point(61) = {26.05, 26.55, 0, lc2};
Point(62) = {24.8, 25.7, 0, lc2};
Point(63) = {24.05, 25.05, 0, lc2};
Point(64) = {23.6, 24.25, 0, lc2};
Point(65) = {23.25, 23.35, 0, lc2};
Point(66) = {23.05, 22.45, 0, lc2};
Point(67) = {22.95, 21.45, 0, lc2};
Point(68) = {22.85, 20.45, 0, lc2};
Point(69) = {22.85, 19.55, 0, lc2};
Point(70) = {22.85, 18.6, 0, lc2};
Point(71) = {22.95, 17.65, 0, lc2};
Point(72) = {23.15, 16.7, 0, lc2};
Point(73) = {23.55, 15.8, 0, lc2};
Point(74) = {24.15, 15.05, 0, lc2};
Point(75) = {24.9, 14.45, 0, lc2};
Point(76) = {25.7, 13.8, 0, lc2};
Point(77) = {26.4, 13.2, 0, lc2};
Point(78) = {27.15, 12.6, 0, lc2};
Point(79) = {28, 12.25, 0, lc2};
Point(80) = {28.85, 12.05, 0, lc2};
Point(81) = {29.75, 11.85, 0, lc2};
Point(82) = {30.7, 11.8, 0, lc2};
Point(83) = {x1_pad_one, 11.85, 0, lc2};
Point(84) = {32.55, 11.9, 0, lc2};
Point(85) = {33.45, 12.1, 0, lc2};
Point(86) = {34.3, 12.3, 0, lc2};
Point(87) = {35.2, 12.75, 0, lc2};
Point(88) = {36, 13.3, 0, lc2};
Point(89) = {36.85, 13.9, 0, lc2};
Point(90) = {37.55, 14.55, 0, lc2};
Point(91) = {38.35, 15.1, 0, lc2};


// bounding box
Point(92) = {x0, y0, 0, lc};
Point(93) = {x1, y0, 0, lc};
Point(94) = {x1, y1, 0, lc};
Point(95) = {x0, y1, 0, lc};

// pad one
Point(96) = {x0_pad_one, y0_pad_one, 0, lc};
Point(97) = {x1_pad_one, y1_pad_one, 0, lc};
Point(98) = {x2_pad_one, y2_pad_one, 0, lc};
Point(99) = {x3_pad_one, y3_pad_one, 0, lc};
Point(100) = {x4_pad_one, y4_pad_one, 0, lc};
Point(101) = {x5_pad_one, y5_pad_one, 0, lc};
Point(102) = {x6_pad_one, y6_pad_one, 0, lc};
Point(103) = {x7_pad_one, y7_pad_one, 0, lc};

// pad two
Point(104) = {x0_pad_two, y0_pad_two, 0, lc};
Point(105) = {x1_pad_two, y1_pad_two, 0, lc};
Point(106) = {x2_pad_two, y2_pad_two, 0, lc};
Point(107) = {x3_pad_two, y3_pad_two, 0, lc};
Point(108) = {x4_pad_two, y4_pad_two, 0, lc};
Point(109) = {x5_pad_two, y5_pad_two, 0, lc};
Point(110) = {x6_pad_two, y6_pad_two, 0, lc};
Point(111) = {x7_pad_two, y7_pad_two, 0, lc};

// bounding box lines
Line(1) = {92, 93};
Line(2) = {93, 94};
Line(3) = {94, 95};
Line(4) = {95, 92};


// keloid boundary (spline)
Spline(5) = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};

//+
Split Curve(5) {55, 83, 36, 15};


// Measurement zone edges : length H_msr

Point(112) = {x1_pad_one, H0*0.5 + H_msr,0 , lc};
Point(113) = {x1_pad_two, H0*0.5 + H_msr,0 , lc};
Point(114) = {x1_pad_one, H0*0.5 - H_msr,0 , lc};
Point(115) = {x1_pad_two, H0*0.5 - H_msr,0 , lc};

Point(116) = {x0_pad_one_square, y0_pad_one_square, 0, lc};
Point(117) = {x1_pad_one_square, y1_pad_one_square, 0, lc};
Point(118) = {x2_pad_one_square, y2_pad_one_square, 0, lc};
Point(119) = {x3_pad_one_square, y3_pad_one_square, 0, lc};

Point(120) = {x0_pad_two_square, y0_pad_two_square, 0, lc};
Point(121) = {x1_pad_two_square, y1_pad_two_square, 0, lc};
Point(122) = {x2_pad_two_square, y2_pad_two_square, 0, lc};
Point(123) = {x3_pad_two_square, y3_pad_two_square, 0, lc};

//+
Recursive Delete {
  Curve{7}; Curve{9}; 
}
//+
Line(9) = {102, 103};
//+
Line(10) = {103, 96};
//+
Line(11) = {96, 97};
//+
Line(12) = {97, 114};
//+
Line(13) = {114, 83};
//+
Line(14) = {83, 98};
//+
Line(15) = {98, 99};
//+
Line(16) = {99, 100};
//+
Line(17) = {100, 101};
//+
Line(18) = {101, 55};
//+
Line(19) = {55, 112};
//+
Line(20) = {112, 102};
//+
Line(21) = {119, 118};
//+
Line(22) = {118, 117};
//+
Line(23) = {117, 116};
//+
Line(24) = {116, 119};
//+
Line(25) = {113, 36};
//+
Line(26) = {36, 109};
//+
Line(27) = {109, 108};
//+
Line(28) = {108, 107};
//+
Line(29) = {107, 106};
//+
Line(30) = {106, 15};
//+
Line(31) = {15, 115};
//+
Line(32) = {115, 105};
//+
Line(33) = {105, 104};
//+
Line(34) = {104, 111};
//+
Line(35) = {111, 110};
//+
Line(36) = {110, 113};
//+
Line(37) = {122, 123};
//+
Line(38) = {123, 120};
//+
Line(39) = {120, 121};
//+
Line(40) = {121, 122};
//+
Line(41) = {112, 113};
//+
Line(42) = {114, 115};
//+
Curve Loop(1) = {6, -18, -17, -16, -15, -14, 8, -30, -29, -28, -27, -26};
//+
Curve Loop(2) = {40, 37, 38, 39};
//+
Curve Loop(3) = {22, 23, 24, 21};
//+
Plane Surface(1) = {1, 2, 3};
//+
Curve Loop(4) = {41, 25, 6, 19};
//+
Plane Surface(2) = {4};
//+
Curve Loop(5) = {8, 31, -42, 13};
//+
Plane Surface(3) = {5};
//+
Curve Loop(6) = {4, 1, 2, 3};
//+
Curve Loop(7) = {10, 11, 12, 42, 32, 33, 34, 35, 36, -41, 20, 9};
//+
Plane Surface(4) = {6, 7};
//+
Physical Surface("keloid_measure", 10) = {1};
//+
Physical Surface("healthy_measure", 20) = {2, 3};
//+
Physical Surface("healthy", 30) = {4};
//+
Physical Curve("pad_one", 1) = {9, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 21, 24, 23};
//+
Physical Curve("pad_two", 2) = {35, 36, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 40, 39, 38};
//+
Physical Curve("pad_one_sensor", 3) = {22};
