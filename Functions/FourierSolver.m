function FourierSolver(randX,randY,date,dataname)

X.X_i = randX;
Y.Y_j = randY;

X.sampleNum = length(randX(1,:));
Y.sampleNum = length(randY(1,:));

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

f.num_samples = 1e4;

hyper.mu = 0;
hyper.nu = 0.01;

hyper.iter = 1e5;

LearnedValues = InverseFunction(X,Y,f,hyper)

save(['/scratch/jmg1030/InverseProblems/data/' date '_' dataname '.mat'],'-struct','LearnedValues','-v7.3');

end