function P = InverseFunction(X,Y,f,hyper)

X_i = X.X_i;
Y_j = Y.Y_j;

X_i = X.X_i(:,randi(length(X.X_i(1,:)),[X.sampleNum,1]));
Y_j = Y.Y_j(:,randi(length(Y.Y_j(1,:)),[Y.sampleNum,1]));

dim = length(X_i(:,1));

% integrating space
dz = 0.01;
z = min(min(X.X_i,Y.Y_j)):0.01:max(max(X.X_i,Y.Y_j));
z = z.*ones(length(X_i(:,1)),1);

% Calculating histogram for Loss Function
Xdist = hist(X.X_i,z);
Xdist = Xdist./(sum(Xdist));

Ydist = hist(Y.Y_j,z);
Ydist = Ydist./(sum(Ydist));

% Integrate given posterior
p_y_x = f.p_y_x(z)./(sum(f.p_y_x(z)));
P_y_x = cumsum(p_y_x);

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
Y_j_Tilde = Sample(P_y_x,z,f.num_samples);

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
        
        dLdBx(:,k,:) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,Y_j_Tilde,num_F,p0_X,k,'x',length(X_i(:,1)));
        dLdBy(:,k,:) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,Y_j_Tilde,num_F,p0_X,k,'y',length(Y_j(:,1)));

    end
    
    % Using a momentum based learning rule, set mu = 0 for normal gradient
    % descent
    Bx = Bx + mu.*Gradx + nu.*dLdBx;
    By = By + mu.*Grady + nu.*dLdBy;
        
    % calculating previous gradient for momentum
    Gradx = dLdBx;
    Grady = dLdBy;
    
    disp(['Mean dLdx: ' num2str(mean(mean(dLdBx)))]);
    disp(['Mean dLdy: ' num2str(mean(mean(dLdBy)))]);
    
    % calculating prediction to check loss
    
    if dim == 1
    YdistPred = conv(Xdist,f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))./sum(f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))),'same');
    
    Loss(II) = sum((YdistPred - Ydist).^2);
        
    disp(['Loss: ' num2str(Loss(II))]);
    end
    
    if mod(II,100) == 0
        
        X_i = X.X_i(:,randi(length(X.X_i(1,:)),[X.sampleNum,1]));
        Y_j = Y.Y_j(:,randi(length(Y.Y_j(1,:)),[Y.sampleNum,1]));
        
    end
    
end

P.Bx = Bx;
P.By = By;

P.z = z;

P.Xdist = Xdist;
P.Ydist = Ydist;

P.YdistPred0 = conv(Xdist,f.p_y_x(z),'same');
P.YdistPred  = conv(Xdist,f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))./sum(f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))),'same');

P.p0_ = p0_;

P.p_y_x = p_y_x;
P.p_y_x_ = f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F))./sum(f.p_y_x(z).*S(FunctionSummation(By,F,z,num_F)));

P.Phi_X = S(FunctionSummation(Bx,F,z,num_F));
P.Phi_Y = S(FunctionSummation(By,F,z,num_F));

P.Loss = Loss;

end