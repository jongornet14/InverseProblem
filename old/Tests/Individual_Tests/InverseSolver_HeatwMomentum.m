repopath = '~/Documents/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e4);
Y.Y_j = normrnd(0,10,[1,1e4]);

X.sampleNum = 1e4;
Y.sampleNum = 1e4;

f.f1 = @(z,k) cos(k.*z);
f.f2 = @(z,k) 0;
f.F  = @(B1,B2,z,k) B1.*f.f1(z,k) + B2.*f.f2(z,k);
f.num_F = 4;

f.S = @(z) log(1 + exp(z));
f.s = @(z) exp(z)./(1 + exp(z));

% Beta terms
f.Bx = zeros(1,f.num_F,2);
f.By = zeros(1,f.num_F,2);

f.p_y_x = @(z) (1./(2.*pi)).*exp(-z.^2./2);

f.num_samples = 100;

hyper.mu = 0;
hyper.nu = 0.1;

hyper.iter = 1e3;

P = InverseFunction(X,Y,f,hyper)

save(
%makeFigure(P)

%NiceSave('Heat_DistributionWMomentum','/Users/jonathangornet/Documents/InverseSolver/Figures','Normal')