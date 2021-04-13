a = 1;
t=1;
MatHtb[kx_,ky_] := -2 t {{0, Cos[kx a/2], Cos[ky a /2]}, {Cos[kx a/2], 0, 
     0}, {Cos[ky a/2], 0, 0}};
Plot[-Im@Tr@Inverse[MatHtb[kx, kx] + 0.01 I IdentityMatrix[3]], {kx, 0, 2 Pi}, PlotRange -> All, AxesOrigin -> {0, 0}];