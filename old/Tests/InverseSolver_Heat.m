repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e3);
Y.Y_j = normrnd(0,10,[1,1e3]);

X.sampleNum = 1e3;
Y.sampleNum = 1e3;

f.f1 = @(z,k) z.^(2.*(k-1));
f.f2 = @(z,k) z.^(2.*k-1);
f.F  = @(B1,B2,z,k) B1.*f.f1(z,k) + B2.*f.f2(z,k);
f.num_F = 4;

f.S = @(z) max(0,z) + 1e-3;%log(1 + exp(z));
f.s = @(z) heaviside(z);%1./(1 + exp(-z));

load('BxVals.mat')
load('ByVals.mat')

% Beta terms
f.Bx = Bx;%ones(1,f.num_F,2);
f.By = By;%ones(1,f.num_F,2);

f.p_y_x = @(z,M) (1./sqrt(2.*pi)).*exp(-(z-M).^2./2);

f.num_samples = 1e2;

hyper.mu = 0;
hyper.nu = 1;

hyper.iter = 1e3;

hyper.savepath = '/Users/jonathangornet/Documents/GitHub/InverseSolver/Tests/';

%%
P = InverseFunctionOptim(X,Y,f,hyper)

%%
makeFigure(P)

%%

f.Bx = Bx;%-10.*ones(1,f.num_F,2);
f.By = By;%-10.*ones(1,f.num_F,2);

P = InverseFunctionPrediction(X,Y,f,hyper)