function vol2= volWarp(vol1,u,v,w,method)

if nargin<5
    method='linear';
end

[x,y,z]=meshgrid(1:size(vol1,2),1:size(vol1,1),1:size(vol1,3));
xu=min(max(single(x+u),1),size(vol1,2));
clear x;
clear u;
yv=min(max(single(y+v),1),size(vol1,1));
clear y;
clear v;
zw=min(max(single(z+w),1),size(vol1,3));
clear z;
clear w;

if strcmp(method,'linear')
    vol2=trilinearSingle(single(vol1),xu,yv,zw);
else
    vol2=interp3(vol1,xu,yv,zw,'*nearest');
end

vol2(isnan(vol2))=0;


