clear all; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function:
%   This function performs PRM analysis in Insp/Exp CT images.
% 
% Inputs:
%         Insp CT Image/Mask; Exp CT Image/Mask
% 
% Outputs:
%         PRM results and image.
% 
% Author: Dante PI Capaldi
% Lab: Robarts Research Institute, London, ON, Canada
% Website: http://www.imaging.robarts.ca/~gep/
% Date: November 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%% PRM Analysis
% Initializing matrices.
exp_mask = [];
exp = [];
insp_mask = [];
insp = [];
PRM_matrix = [];
PRM_matrix_1 = [];
PRM_matrix_2 = [];
PRM_matrix_3 = [];
PRM_matrix_4 = [];
tot = 0;
tot_1 = 0;
tot_2 = 0;
tot_3 = 0;
tot_4 = 0;

% Read Expiration CT Masks
file_1 = load_nii('ExpLungMask.nii');
exp_mask = file_1.img;
exp_mask(exp_mask > 11) = 1;
exp_mask(exp_mask == 10) = 0;
exp_mask = double(exp_mask);
% Read Expiration CT Image
file_2 = load_nii('ExpLung.nii');
exp = file_2.img;
% To show ct down below
exp_tmp = exp;
exp = double(exp).*exp_mask;

% Read Warped Inspiratory CT Mask
file_3 = load_nii('InspLungMask_def.nii');
insp_mask = file_3.img;
insp_mask(insp_mask > 11) = 1;
insp_mask(insp_mask == 10) = 0;
insp_mask = double(insp_mask);
% Read Warped Inspiratory CT Image
file_4 = load_nii('InspLung_def.nii');
insp = file_4.img;
insp = double(insp).*exp_mask;
insp_tmp = insp;

% Standard Thresholds
thresh1 = -950; % Inspiration Threshold
thresh2 = -856; % Expiration Threshold

% 1,2,10,20 are arbitrary values chosen so that when insp and exp are
%   summed together, they will produce four distincted result corresponding
%   to the four categories.
insp(insp==0) = NaN;
insp(insp>thresh1) = 1;
insp(insp<=thresh1) = 2;
insp(isnan(insp)) = 0 ;
exp(exp==0) = NaN;
exp(exp>thresh2) = 10;
exp(exp<=thresh2) = 20;
exp(isnan(exp)) = 0 ;
PRM_matrix = insp + exp;

tot = sum(sum(sum(exp_mask)));

% Normal tissue = category1.
PRM_matrix(PRM_matrix==11)=1;
PRM_matrix_1(PRM_matrix==1)=1;
tot_1 = sum(sum(sum(PRM_matrix_1)));
% Non-applicable = category4.
PRM_matrix(PRM_matrix==12)=4;
PRM_matrix_4(PRM_matrix==4)=1;
tot_4 = sum(sum(sum(PRM_matrix_4)));
% Small-airways disease = category2.
PRM_matrix(PRM_matrix==21)=2;
PRM_matrix_2(PRM_matrix==2)=1;
tot_2 = sum(sum(sum(PRM_matrix_2)));
% Emphysema = category3.
PRM_matrix(PRM_matrix==22)=3;
PRM_matrix_3(PRM_matrix==3)=1;
tot_3 = sum(sum(sum(PRM_matrix_3)));

% Normal tissue percentage.
disp(tot_1*100/tot);
% Small-airways disease percentage.
disp(tot_2*100/tot);
% Emphysema percentage.
disp(tot_3*100/tot);


%% For Displaying PRM (need to have imshow3D.m)
pixelDim = [0.781250 1.25];

ct = PRM_matrix;
ct = flipdim(ct,3);
ct1 = permute(ct,[3 1 2]);
ct1 = flipdim(ct1,2);

h = waitbar(0,'Starting...','Name','Resizing 3D CT Matrix:'); 
for i = 1 : int16(size(ct1,3))
    imgtmp = ct1(:,:,i);
    imgtmp = imresize(imgtmp,[round(size(imgtmp,1)*pixelDim(2)) round(size(imgtmp,2)*pixelDim(1))]);
    ct2(:,:,i) = imgtmp;
    perc = double((double(i)/(size(ct1,3))));
    waitbar(perc,h,sprintf('%2.2f%% along...',perc*100))
end
waitbar(1,h,sprintf('%2.2f%% Complete',perc*100))
close(h)


cmap=[0 0 0;...
      0 1 0; ...
      1 1 0; ...
      1 0 0];

ct2 = uint8(ct2);
ct2(ct2==4)=0;
ct_1 = ct2 == 1;
ct_2 = ct2 == 2;
ct_3 = ct2 == 3;
ct_tot = ct_1*1 + ct_2*2 + ct_3*3;

figure,imshow3D(ct_tot);colormap(cmap);
