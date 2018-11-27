function P = InverseFunctionPrediction(X,Y,f,hyper)

X_i = X.X_i;
Y_j = Y.Y_j;

dim = length(X_i(:,1));

% integrating space
z = linspace(min(min(X.X_i,Y.Y_j)),max(max(X.X_i,Y.Y_j)),100);
dz = z(2) - z(1);
z = z.*ones(length(X_i(:,1)),1);

% Calculating histogram for Loss Function
Xdist = hist(X.X_i,z);
Xdist = Xdist./(sum(Xdist));

Ydist = hist(Y.Y_j,z);
Ydist = Ydist./(sum(Ydist));

% Importance Sampling
p0_ = zeros(length(X_i(:,1)),length(z));
for i = 1:length(X_i(1,:))
    for j = 1:length(X_i(:,1))
        p0_(j,:) = p0_(j,:) + (1/length(X_i(1,:))).*exp(-(z(j,:)-X_i(i)).^2);
    end
end
p0_ = p0_./sum(p0_);

% Summation Function
f1 = f.f1;
f2 = f.f2;
F = @(B1,B2,z,k) B1.*f1(z,k) + B2.*f2(z,k);
num_F = f.num_F;

% Beta terms
Bx = f.Bx;
By = f.By;

% Positivity function
S = f.S;
s = f.s;

% Sampling
[X_i_Tilde,p0_X] = Sample(p0_,z,f.num_samples);

P.Bx = Bx;
P.By = By;

P.z = z;

P.Xdist = Xdist;
P.Ydist = Ydist;

P.X_i_Tilde = X_i_Tilde;

P.p0_X = p0_X;

P.p0_ = p0_;

P.p_y_x = f.p_y_x(z,0);
P.p_y_x_ = f.p_y_x(z,0).*S(FunctionSummation(By,F,z,num_F))./sum(f.p_y_x(z,0).*S(FunctionSummation(By,F,z,num_F)));

P.YPred0(1,:) = z;
P.YPred(1,:) = z;

P.YPred0(2,:) = conv(Xdist,P.p_y_x,'same');
P.YPred(2,:) = conv(Xdist,P.p_y_x_,'same');

% P.YPred0 = Prediction(zeros(length(X_i(:,1)),num_F,2),zeros(length(Y_j(:,1)),num_F,2),f,X_i_Tilde,z,num_F,p0_X,length(X_i(:,1)));
% P.YPred = Prediction(Bx,By,f,X_i_Tilde,z,num_F,p0_X,length(X_i(:,1)));

P.Phi_X = S(FunctionSummation(Bx,F,z,num_F));
P.Phi_Y = S(FunctionSummation(By,F,z,num_F));

figure;
subplot(2,1,1)
plot(z,P.Phi_X);
subplot(2,1,2)
plot(z,P.Phi_Y);

makeFigure(P);

end