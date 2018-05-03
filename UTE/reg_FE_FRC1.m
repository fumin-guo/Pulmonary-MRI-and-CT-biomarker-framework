clc
clear all
close all
tic

sub = 'your directory/'

img1 = metaImageRead([sub '/FRC1.mhd']);
img1_hdr = metaImageInfo([sub '/FRC1.mhd']);
img1 = double(img1);
img1 = (img1 - min(img1(:)))/(max(img1(:)) - min(img1(:)))*100;


img2 = metaImageRead([sub1 '/FE.mhd']);
img2 = double(img2);
img2_orig = img2 ;
img2 = (img2 - min(img2(:)))/(max(img2(:)) - min(img2(:)))*100;

numIter = 200;
steps = 0.10; 
alpha = 5;
cc = 0.25;

Ux = zeros(2,2,2,'single'); 
Uy = Ux; 
Uz = Ux;
wi = 0.5;

errbound = 2e-4;
warps = [4 4 2];
levels = [4 2 1];
offset = 1;

tic

for j=1:length(levels)
    
    hs=fspecial('gaussian',[5,1],levels(j)/max(levels(:))/2);
    
    img1_l = volresize(volfilter(img1,hs),ceil(size(img1)./levels(j)));
    img2_l = volresize(volfilter(img2,hs),ceil(size(img2)./levels(j)));
    
    % resize volumes for current level
    
    [Ux,Uy,Uz] = resizeFlow(Ux,Uy,Uz,size(img1_l));
    
    ux = zeros(ceil(size(img1)./levels(j)));
    uy = zeros(ceil(size(img1)./levels(j)));
    uz = zeros(ceil(size(img1)./levels(j)));
    
    [n_rows,n_cols,n_heights]=size(img1_l);
    
    for k = 1:warps(j)
        
        Ux = Ux + ux*wi;
        Uy = Uy + uy*wi;
        Uz = Uz + uz*wi;        
        
        img1_w = img1_l;
        img2_w = volWarp(img2_l,Ux,Uy,Uz);
        
        [~,ssc_q1] = SSC_descriptor_H(img1_w,0.5,offset);
        [~,ssc_q2] = SSC_descriptor_H(img2_w,0.5,offset);
        
        gt = hammingDist3D(ssc_q2, ssc_q1);

        gx = (hammingDist3D(volshift(ssc_q2,1,0,0), ssc_q1)-hammingDist3D(volshift(ssc_q2,-1,0,0), ssc_q1))/2;
        gy = (hammingDist3D(volshift(ssc_q2,0,1,0), ssc_q1)-hammingDist3D(volshift(ssc_q2,0,-1,0), ssc_q1))/2;
        gz = (hammingDist3D(volshift(ssc_q2,0,0,1), ssc_q1)-hammingDist3D(volshift(ssc_q2,0,0,-1), ssc_q1))/2;
               
        gx(:,[1,n_cols],:)=0;
        gy([1,n_rows],:,:)=0;
        gz(:,:,[1,n_heights])=0;
        
        gf = gx.*gx + gy.*gy + gz.*gz;
        
        
        % - para: a sequence of parameters for the algorithm
        %   para[0,1,2]: rows, cols, heights of the given image
        %   para[3]: the maximum iteration number
        %   para[4]: the error bound for convergence
        %   para[5]: cc for the step-size of augmented Lagrangian method
        %   para[6]: the step-size for the graident-projection step to the
        %   total-variation function.
        varaParas = [n_rows; n_cols; n_heights; levels(j)*numIter; errbound; cc; steps; alpha];
        
        % CPU based flow adpating
        % [ux, uy, uz, err, num, timet] = Reg_TVL1_Newton_mex(single(varaParas), single(Ux), single(Uy), single(Uz), single(gx), single(gy), single(gz), single(gt), single(gf));
        
        % GPU based flow adpating
        [ux, uy, uz, err, num, timet] = Reg_TVL1_Newton_GPU_opt(single(varaParas), single(Ux), single(Uy), single(Uz), single(gx), single(gy), single(gz), single(gt), single(gf));
        
    end
    %     % testing
    %     out_vol = volWarp(g2q,Ux,Uy,Uz);
    %     slice=uint16(size(out_vol,3)/2);
    %     figure();
    %     subplot(2,3,1); imshow(g1q(:,:,slice),[]); title('g1q')
    %     subplot(2,3,2); imshow(g2q(:,:,slice),[]); title('g2q')
    %     subplot(2,3,3); imshow(out_vol(:,:,slice),[]); title('g2q_w')
    %     subplot(2,3,4); imshow(abs(g1q(:,:,slice)-out_vol(:,:,slice)),[]); title('abs(g1q-g2q_w)')
    %     subplot(2,3,5); quiver(Uy(:,:,slice),Ux(:,:,slice)); axis('equal'); title('Uy,Ux')
    %     tmp = zeros([size(g1q(:,:,slice)),3]);
    %     tmp(:,:,1) = mat2gray(Uy(:,:,slice)); tmp(:,:,2) = mat2gray(Ux(:,:,slice)); tmp(:,:,3) = mat2gray(Uz(:,:,slice));
    %     subplot(2,3,6); imshow(tmp);

    Ux = Ux + ux*wi;
    Uy = Uy + uy*wi;
    Uz = Uz + uz*wi;
    
end

toc
timet = toc

if(levels(end)>1)
    [Ux,Uy,Uz] = resizeFlow(Ux,Uy,Uz, size(img1));
end

g2_def = volWarp(img2_orig,Ux,Uy,Uz);
metaImageWrite(g2_def,[sub '/FE_def.mhd'],img1_hdr);
