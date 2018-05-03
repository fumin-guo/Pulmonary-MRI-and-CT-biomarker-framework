function vol=volfilter(vol,h,method)

if nargin<3
    method='replicate';
end

h=reshape(h,[numel(h),1,1]);
vol=imfilter(vol,h,method);

h=reshape(h,[1,numel(h),1]);
vol=imfilter(vol,h,method);

h=reshape(h,[1,1,numel(h)]);
vol=imfilter(vol,h,method);
