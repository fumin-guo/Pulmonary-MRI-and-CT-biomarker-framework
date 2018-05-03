function HMRI_segmentation(filename_H_3,filename_H_seeds,dir)

h = fspecial('gaussian',5,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% smooth the image a bit for gradient calculation
H_seeds = metaImageRead(filename_H_seeds);
H_img3D = double(metaImageRead(filename_H_3));
[rows,cols,heights] = size(H_img3D);
H_hdr = metaImageInfo(filename_H_3);

min_gray = min(H_img3D(:));
max_gray = max(H_img3D(:));
H_img3D_norm = (H_img3D-min_gray)/(max_gray-min_gray);

ur1_smooth = imfilter(H_img3D_norm,h,'same');
[dx,dy,dz] = gradient(255*ur1_smooth);
grad = sqrt(dx.^2+dy.^2+dz.^2);
max_grad = max(grad(:));
min_grad = min(grad(:));
grad = (grad-min_grad)./(max_grad-min_grad);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up the regularization term.
% for H MRI segmentation
alpha = 0.14 + 2*exp(-75*grad);
constant = 0.02;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%set up the data cost term.
large_value = 1e5;
scale = 500;
samples = 0:1:scale;
Width = 3;

IND = round(scale*H_img3D_norm)+1;
% label 1
Ivals = H_img3D_norm(find(H_seeds == 1));
Ivals = Ivals*scale;
pdf_lung = ksdensity(Ivals,samples,'Width',Width);

look_lung = -constant*log(pdf_lung./length(Ivals)+eps);
Ct1 = look_lung(IND);
Ct1(H_seeds == 2) = large_value;
Ct1(H_seeds == 3) = large_value;
Ct1(:,:,[1,end]) = large_value;
Ct1(:,[1,end],:) = large_value;
Ct1([1,end],:,:) = large_value;


%label 2
Ivals = H_img3D_norm(find(H_seeds == 2));
Ivals = Ivals*scale;
pdf_lung = ksdensity(Ivals,samples,'Width',Width);

look_lung = -constant*log(pdf_lung./length(Ivals)+eps);
Ct2 = look_lung(IND);
Ct2(H_seeds == 1) = large_value;
Ct2(H_seeds == 3) = large_value;
Ct2(:,:,[1,end]) = large_value;
Ct2(:,[1,end],:) = large_value;
Ct2([1,end],:,:) = large_value;


%label 3
Ivals = H_img3D_norm(find(H_seeds == 3));
Ivals = Ivals*scale;
pdf_lung = ksdensity(Ivals,samples,'Width',Width);

look_lung = -constant*log(pdf_lung./length(Ivals)+eps);
Ct3 = look_lung(IND);
Ct3(H_seeds == 1) = large_value;
Ct3(H_seeds == 2) = large_value;
Ct3(H_seeds == 3) = 0;

Ct(:,:,:,1)= Ct1;
Ct(:,:,:,2)= Ct2;
Ct(:,:,:,3)= Ct3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up the segmentation parameters
cc = 0.25;
nlab = 3;

varParas = [rows; cols; heights; nlab; 300; 1e-4; cc; 0.11];
penalty = alpha;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the segmentation
[u, erriter,num,timet] = CMF3D_ML_GPU(single(penalty), single(Ct), single(varParas));
u = cc*u;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get final labels,
[um,uu] = max(u, [], 4);
uout = uu;
uout(uu==3)=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save the resutls to files.
metaImageWrite(uout,[dir '/H_out.mhd'],H_hdr);
end
