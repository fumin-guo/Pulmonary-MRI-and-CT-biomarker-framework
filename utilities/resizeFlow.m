function [u2,v2,w2]=resizeFlow(u1,v1,w1,newsize)

if sum(newsize==size(u1))==3
    u2=u1; v2=v1; w2=w1;
else
    scalings=single([newsize(1)/size(u1,1),newsize(2)/size(u1,2),newsize(3)/size(u1,3)]);
    
    a=linspace(1,size(u1,2),newsize(2));
    b=linspace(1,size(u1,1),newsize(1));
    c=linspace(1,size(u1,3),newsize(3));
    
    [xi,yi,zi]=meshgrid(a,b,c);
    xi=single(xi);
    yi=single(yi);
    zi=single(zi);
    
    u2=trilinearSingle(single(u1).*scalings(2),xi,yi,zi); 
    u2(isnan(u2))=0; clear u1; 
    v2=trilinearSingle(single(v1).*scalings(1),xi,yi,zi);
    v2(isnan(v2))=0; clear v1;
    w2=trilinearSingle(single(w1).*scalings(3),xi,yi,zi);
    w2(isnan(w2))=0; clear w1;
end
