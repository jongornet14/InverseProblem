function [dW,E,ED] = AdaDelta(E,ED,dL)

g = 0.9;
n = 1e-4;

E = g.*E + (1-g).*dL.^2;

dLl = (n./sqrt(E+1e-8)).*dL;

ED = g.*ED + (1-g).*dLl.^2;

dW = sqrt(ED+1e-8)./sqrt(E+1e-8).*dL;
