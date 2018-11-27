function [dW,mt,vt] = Adam(mt,vt,dL)

b1 = 0.9;
b2 = 0.999;
e = 1e-8;

mt = b1.*mt + (1-b1).*dL;
vt = b2.*vt + (1-b2).*dL.^2;

mt = mt./(1-b1);
vt = vt./(1-b2);

dW = mt./(sqrt(vt)+e);

end
