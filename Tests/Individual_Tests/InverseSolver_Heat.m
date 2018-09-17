repopath = '/scratch/jmg1030/InverseProblems/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e5);
Y.Y_j = normrnd(0,10,[1,1e5]);

X.sampleNum = 1e5;
Y.sampleNum = 1e5;

f.f1 = @(z,k) cos(k.*z);
f.f2 = @(z,k) sin(k.*z);
f.F  = @(B1,B2,z,k) B1.*f.f1(z,k) + B2.*f.f2(z,k);
f.num_F = 4;

f.S = @(z) log(1 + exp(z));
f.s = @(z) exp(z)./(1 + exp(z));

% Beta terms
f.Bx = zeros(1,f.num_F,2);
f.By = zeros(1,f.num_F,2);

f.p_y_x = @(z) 1./sqrt(2.*pi).*exp(-z.^2./2);

f.num_samples = 100;

hyper.mu = 0;
hyper.nu = 0.01;

hyper.iter = 1e5;

LearnedValues = InverseFunction(X,Y,f,hyper)

save(['/scratch/jmg1030/InverseProblems/data/9-17-2018/HeatEq.mat'],'-struct','LearnedValues','-v7.3');