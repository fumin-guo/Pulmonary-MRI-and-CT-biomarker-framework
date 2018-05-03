function [ssc,ssc_q]=SSC_descriptor(I,sigma,delta)
% Calculation of SSC (self-similarity context)
%
% If you use this implementation please cite:
% M.P. Heinrich et al.: "Towards Realtime Multimodal Fusion for 
% Image-Guided Interventions Using Self-similarities"
% MICCAI (2013) LNCS Springer
%
% M.P. Heinrich et al.: "MIND: Modality Independent Neighbourhood
% Descriptor for Multi-Modal Deformable Registration"
% Medical Image Analysis (2012)
%
% Contact: heinrich(at)imi(dot)uni-luebeck(dot)de
%
% I: input volume (3D)
% sigma: Gaussian weighting for patches 
% delta: Distance between patch centres
%
% ssc: output descriptor (4D)
% ssc_q: quantised descriptor (3D) uint64
% important: ssc_q requires the compilitation of quantDescriptor.cpp

I=single(I);
if nargin<2
    sigma=0.8;
end
if nargin<3
    delta=1;
end

% Filter for efficient patch SSD calculation
filt=fspecial('gaussian',[ceil(sigma*3/2)*2+1,1],sigma);

%displacements between patches
dx=[+1,+1,-1,+0,+1,+0].*delta;
dy=[+1,-1,+0,-1,+0,+1].*delta;
dz=[+0,+0,+1,+1,+1,+1].*delta;

sx=[-1,+0,-1,+0,+0,+1,+0,+0,+0,-1,+0,+0].*delta;
sy=[+0,-1,+0,+1,+0,+0,+0,+1,+0,+0,+0,-1].*delta;
sz=[+0,+0,+0,+0,-1,+0,-1,+0,-1,+0,-1,+0].*delta;

% Self-similarity Distanceskuw
distances=zeros([size(I),numel(dx)],'single');
I_dummy = zeros([size(I),numel(dx)],'single');

% Calculating Gaussian weighted patch SSD using convolution
for i=1:numel(dx)
    distances(:,:,:,i)=volfilter((I-volshift(I,dx(i),dy(i),dz(i))).^2,filt);
    % this is adde by Fumin to calculate the global variance instead of \sigma
    I_dummy(:,:,:,i) = volshift(I,dx(i),dy(i),dz(i));
end

% Shift 'second half' of distances to avoid redundant calculations
index=[7,7,8,8,9,9,10,10,11,11,12,12]-6;
ssc=zeros([size(I),numel(index)],'single');
for i=1:numel(index)
    tem = distances(:,:,:,index(i));
    ssc(:,:,:,i)=volshift(tem,sx(i),sy(i),sz(i));
end
clear distances;

% Remove minimal distance to scale descriptor to max=1
matrixmin=min(ssc,[],4);
for i=1:numel(index)
    ssc(:,:,:,i)=ssc(:,:,:,i)-matrixmin;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Variance measure (standard absolute deviation)
V=mean(ssc,4);
val1=[0.01*(mean(V(:))),100*mean(V(:))];
V=min(max(V,min(val1)),max(val1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this is adde by Fumin to calculate the global variance instead of \sigma
% V = mean(I_dummy,4);
% V = ((7/6)^0.5*(I - V)).^2;
% V = mean(V(:));

% descriptor calculation according
for i=1:numel(index)
    ssc(:,:,:,i)=exp(-ssc(:,:,:,i)./1);
end

if nargout>1
    % quantise descriptors into uint64
    ssc_q=quantDescriptor(ssc,8);

end
end
