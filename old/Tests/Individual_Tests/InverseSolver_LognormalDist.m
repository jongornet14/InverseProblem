repopath = '~/Documents/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e3);
Y.Y_j = lognrnd(0,1,[1,1e3]);

X.sampleNum = 100;
Y.sampleNum = 100;

f.fk = @(z,k) sin(k.*z);
f.num_F = 20;

f.S = @(z) log(1 + exp(z));
f.s = @(z) exp(z)./(1 + exp(z));

% Beta terms
f.Bx = zeros(1,20);
f.By = zeros(1,20);

f.p_y_x = @(z) (1./(2.*pi)).*exp(-z.^2./2);

f.num_samples = 100;

hyper.mu = 0;%0.01;
hyper.nu = 0.01;

hyper.iter = 1e4;

P = InverseFunction(X,Y,f,hyper)

makeFigure(P)

NiceSave('Lognormal_Distribution','/Users/jonathangornet/Documents/InverseSolver/Figures','Normal')