function P = InverseFunctionOptim(X,Y,f,hyper)

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

nuX = nu.*ones(length(X_i(:,1)),num_F,2);
nuY = nu.*ones(length(Y_j(:,1)),num_F,2);

% Set initial gradient to 0
Gradx = zeros(length(X_i(:,1)),num_F,2);
Grady = zeros(length(Y_j(:,1)),num_F,2);

% Loss function
Loss = zeros(1,hyper.iter);

dLdBx = zeros(length(X_i(:,1)),num_F,2);
dLdBy = zeros(length(Y_j(:,1)),num_F,2);

BRange = -100:100;
BLength = length(BRange);

dBx = zeros(length(X_i(:,1)),num_F,2,BLength);
dBy = zeros(length(Y_j(:,1)),num_F,2,BLength);

for II = 1:BLength
    
    Bx = BRange(II).*ones(length(X_i(:,1)),num_F,2);
    By = BRange(II).*ones(length(Y_j(:,1)),num_F,2);
    
    % Looping through each beta term to find its gradient vector
    parfor k = 1:num_F
        
        dBx(:,k,:,II) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,z,num_F,p0_X,k,'x',length(X_i(:,1)));
        dBy(:,k,:,II) = Gradient(Bx,By,f,X_i,Y_j,X_i_Tilde,z,num_F,p0_X,k,'y',length(Y_j(:,1)));

    end
   
end

%%

P.dBx = dBx;
P.dBy = dBy;

P.Bx = Bx;
P.By = By;

end