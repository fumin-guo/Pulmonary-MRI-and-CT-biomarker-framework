function vol2=volresize(vol1,newsize,method)


if sum(newsize==size(vol1))==3
    vol2=vol1;
else
    
a=single(linspace(1,size(vol1,1),newsize(1)));
b=single(linspace(1,size(vol1,2),newsize(2)));
c=single(linspace(1,size(vol1,3),newsize(3)));

[xi,yi,zi]=meshgrid(b,a,c);

if nargin<3
    vol2=trilinearSingle(single(vol1),single(xi),single(yi),single(zi));

else
    
    vol2=interp3(single(vol1),xi,yi,zi,method);
end


vol2(isnan(vol2))=0;
vol2=single(vol2);

end