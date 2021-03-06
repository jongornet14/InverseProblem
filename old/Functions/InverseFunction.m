function P = InverseFunction(X,Y,f,hyper)

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
P0_ = cumsum(p0_);

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
[X_i_Tilde,p0_X] = Sample(P0_,z,f.num_samples);

% Gradient terms
mu = hyper.mu;
nu = hyper.nu;

% Set initial gradient to 0
Gradx = 0;
Grady = 0;

% Loss function
Loss = zeros(1,hyper.iter);

dLdBx = zeros(length(X_i(:,1)),num_F,2);
dLdBy = zeros(length(Y_j(:,1)),num_F,2);

for II = 1:hyper.iter
    
    % Looping through each beta term to find its gradient vector
    parfor k = 1:num_F
        
        dLdBx(:,k,:) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,z,num_F,p0_X,k,'x',length(X_i(:,1)));
        dLdBy(:,k,:) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,z,num_F,p0_X,k,'y',length(Y_j(:,1)));

    end
    
    % Using a momentum based learning rule, set mu = 0 for normal gradient
    % descent
    Bx = Bx + mu.*Gradx + nu.*dLdBx;
    By = By + mu.*Grady + nu.*dLdBy;
        
    % calculating previous gradient for momentum
    Gradx = dLdBx;
    Grady = dLdBy;
    
    % calculating prediction to check loss
    
    YPred = Prediction(Bx,By,f,X_i_Tilde,z,num_F,p0_X,length(X_i(:,1)));    
    Loss(II) = Cost(Ydist,YPred(2,:),dim);   
    
    if mod(II, hyper.iter/10) == 0
        
        disp([num2str(round(100.*II./hyper.iter)),'% Done!']);
        disp(['Loss: ' num2str(Loss(II))]);
        
        disp(['Mean dLdx: ' num2str(mean(mean(dLdBx)))]);
        disp(['Mean dLdy: ' num2str(mean(mean(dLdBy)))]);
        
        [X_i_Tilde,p0_X] = Sample(P0_,z,f.num_samples);
        
    end
    
end

P.Bx = Bx;
P.By = By;

P.z = z;

P.Xdist = Xdist;
P.Ydist = Ydist;

P.X_i_Tilde = X_i_Tilde;

P.p0_X = p0_X;

YPred = Prediction(Bx,By,f,X_i_Tilde,Y_j_Tilde,num_F,p0_X,length(X_i(:,1)));
YdistPred = hist(YPred,z);
YdistPred = YdistPred./sum(YdistPred);

P.YdistPred0 = conv(Xdist,f.p_y_x(z),'same');
P.YdistPred  = YdistPred;

P.p0_ = p0_;

P.p_y_x = p_y_x;
P.p_y_x_ = f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))./sum(f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F)));

P.Phi_X = S(FunctionSummation(Bx,F,z,num_F));
P.Phi_Y = S(FunctionSummation(By,F,z,num_F));

P.Loss = Loss;

end