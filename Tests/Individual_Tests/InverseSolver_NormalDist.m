repopath = '~/Documents/InverseSolver';
addpath(genpath(repopath));

X.X_i = randn(1,1e4);
Y.Y_j = randn(1,1e4);

X.sampleNum = 1e4;
Y.sampleNum = 1e4;

f.f1 = @(z,k) cos(k.*z);
f.f2 = @(z,k) sin(k.*z);
f.F  = @(B1,B2,z,k) B1.*f.f1(z,k) + B2.*f.f2(z,k);
f.num_F = 4;

f.S = @(z) log(1 + exp(z));
f.s = @(z) 1./(1 + exp(-z));

% Beta terms
f.Bx = zeros(1,f.num_F,2);
f.By = zeros(1,f.num_F,2);

f.p_y_x = @(z,M) (1./(2.*pi)).*exp(-(z-M).^2./2);

f.num_samples = 1e2;

hyper.mu = 0;
hyper.nu = 0.01;

hyper.iter = 1e3;

P = InverseFunction(X,Y,f,hyper)

%%
makeFigure(P)

%%
bins = linspace(-10,10,100);
mapX = hist(P.X_i_Tilde,bins);
mapY = hist(P.Y_j_Tilde,bins);

figure
plot(bins,mapX,'r')
hold on
plot(bins,mapY,'b')

%%

YY = Prediction(P.Bx,P.By,f,P.X_i_Tilde,P.Y_j_Tilde,f.num_F,P.p0_X,length(X.X_i(:,1)));

figure
plot(YY(1,:),YY(2,:),'.k')

% NiceSave('Normal_Distribution','/Users/jonathangornet/Documents/InverseSolver/Figures','Normal')