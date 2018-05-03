clc
clear all
close all

FE_si=metaImageRead('FE_lung_def.mhd');
FRC_si=metaImageRead('FRC_lung_def.mhd');
FRC1_si=metaImageRead('FRC1_lung.mhd');
FI_si=metaImageRead('FI_lung_def.mhd');

[rows, cols, slices] = size(FE_si);

r=zeros(rows, cols, slices);
m=zeros(rows, cols, slices);
b=zeros(rows, cols, slices);

for i=1:rows
    for j=1:cols
        for k=1:slices
            SIpoints=[FE_si(i,j,k),FRC_si(i,j,k), FRC1_si(i,j,k),FI_si(i,j,k)];
            
	    % the four values below are the volume of lung ottained from segmentation
            [r(i,j,k),m(i,j,k),b(i,j,k)]= regression([5.58,5.72,6.37,7.04],SIpoints);
        end
    end
end

% display the obtained dynamic proton density (slop) map.
imshow3D(m)            
