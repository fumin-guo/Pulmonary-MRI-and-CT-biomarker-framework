function vol1shift=volshift(vol1,x,y,z)

[m,n,o,p]=size(vol1);

vol1shift=zeros(size(vol1));

x1s=max(1,x+1); x2s=min(n,n+x);
y1s=max(1,y+1); y2s=min(m,m+y);
z1s=max(1,z+1); z2s=min(o,o+z);

x1=max(1,-x+1); x2=min(n,n-x);
y1=max(1,-y+1); y2=min(m,m-y);
z1=max(1,-z+1); z2=min(o,o-z);

vol1shift(y1:y2,x1:x2,z1:z2,:)=vol1(y1s:y2s,x1s:x2s,z1s:z2s,:);
end