repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e3);
Y.Y_j = randn(1,1e3);

X.sampleNum = 1e3;
Y.sampleNum = 1e3;

f.f1 = @(z,k) cos(k.*z);
f.f2 = @(z,k) sin(k.*z);
f.F  = @(B1,B2,z,k) B1.*f.f1(z,k) + B2.*f.f2(z,k);
f.num_F = 4;

f.S = @(z) log(1 + exp(z));
f.s = @(z) 1./(1 + exp(-z));

load('BxVals.mat')
load('ByVals.mat')

% Beta terms
f.Bx = Bx;%zeros(1,f.num_F,2);
f.By = By;%zeros(1,f.num_F,2);

f.p_y_x = @(z,M) (1./sqrt(2.*pi)).*exp(-(z-M).^2./2);

f.num_samples = 1e2;

hyper.mu = 0;
hyper.nu = 0.001;

hyper.iter = 1e3;

%%
P = InverseFunctionOptim(X,Y,f,hyper)

%%
makeFigure(P)

%%

P = InverseFunctionPrediction(X,Y,f,hyper)