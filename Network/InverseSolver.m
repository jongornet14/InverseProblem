function [vals] = InverseSolver(params)

repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

vecField = 6;
hidden_num = 6;

%A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1));z.^(transpose(0:vecField-1));exp(z.*transpose(-2:2))];
%A  = @(z) [exp(-z.^2./transpose(1:vecField));z.^transpose(0:vecField-1);exp(z.*transpose(-2:2))];
%A  = @(z) [exp(-z.^2./transpose(1:vecField));z.^transpose(0:vecField-1);exp(z.*transpose(0:vecField-1))];
%A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1))];
A = @(x) [ cos(0.*x); 2.*x; 4.*x.^2 - 2; 8.*x.^3 - 12.*x; 16.*x.^4 - 48.*x.^2 + 12; 32.*x.^5 - 160.*x.^3 + 120.*x];

% F  = @(z) exp(z)+1e-8;
% f  = @(z) exp(z)+1e-8;

F = @(z) z.^2;
f = @(z) 2.*z;

% F = @(z) max(z,0)+1e-8;
% f = @(z) heaviside(z)+1e-8;

% F = @(z) log(1+exp(z))+1e-8;
% f = @(z) 1./(1+exp(-z))+1e-8;

% F = @(z) atan(z);
% f = @(z) 1./(1+z.^2);

x_i = params.x_i;
y_j = params.y_j;

fx_i = params.fx_i;
fy_j = params.fy_j;

%fF = params.fF;

p_x_i_ = zeros(1,length(x_i));
p_y_j_ = zeros(1,length(x_i));

x_i_ = zeros(1,length(x_i));
y_j_ = zeros(1,length(x_i));

z = -10:0.01:10;
Z = transpose(z);
dZ = mean(diff(Z));

r0 = sum(exp(-(Z-x_i).^2),2);
r0 = r0./sum(r0);

for ii = 1:length(x_i)
    
    r = rand;
    
    [c,index] = min(abs(r-cumsum(r0)));
    x_i_(ii) = Z(index);
    p_x_i_(ii) = r;
    
end
    
P_Y_X = conv(TransitionFunction(linspace(0,10,length(x_i)),0),r0);
P_Y_X = P_Y_X(1:length(Z)).*dZ;

Y = NyquistSample(Z,P_Y_X);

R = randi([1 length(Y)],[1 1e4]);
y_j_ = Y(R);

iter = params.iter;

Gradx = nan(hidden_num,iter);
Grady = nan(hidden_num,iter);

% mapX = hist(x_i,z);
% mapY = hist(y_j,z);
% mapZ = hist(y_j_,z);
% 
% mapX = movmean(mapX./sum(mapX),10);
% mapY = movmean(mapY./sum(mapY),10);
% mapZ = movmean(mapZ./sum(mapZ),10);

% figure
% subplot(3,1,1)
% plot(z,mapX)
% subplot(3,1,2)
% plot(z,mapY)
% subplot(3,1,3)
% plot(z,mapZ)

Loss = zeros(1,iter);

%%

if isfield(params,'Wx')
Wx = params.Wx;
else
Wx = randn(1,hidden_num); %x values
end
if isfield(params,'Wy')
Wy = params.Wy;
else
Wy = randn(1,hidden_num); %y values
end

Ex = zeros(1,hidden_num);
Ey = zeros(1,hidden_num);

EDx = zeros(1,hidden_num);
EDy = zeros(1,hidden_num);

predictions = zeros(10,length(z));

a = 1;
b = 0;
w = zeros(hidden_num,1);

%%

disp('Starting Training...');

Dx = ones(1,hidden_num);
Dy = ones(1,hidden_num);

for epoch = 1:iter
    
    Rx = randi([1 length(x_i)],[1 1e4]);
    Ry = randi([1 length(y_j)],[1 1e4]);
    
    Rx_ = randi([1 length(x_i_)],[1 1e4]);
    Ry_ = randi([1 length(y_j_)],[1 1e4]);

    if mod(epoch,iter/10) == 0
        p_y_x = F(Wy*A(z)).*(TransitionFunction(z,0))./sum(F(Wy*A(z)).*(TransitionFunction(z,0)));
        pred = conv(fx_i,p_y_x,'same');
        disp(['Loss: ' num2str(Cost(fy_j./sum(fy_j),pred./sum(pred),1))]);
        disp([num2str(round(100.*epoch./iter)),'% Done!'])
        Loss(epoch) = Cost(fy_j./sum(fy_j),pred./sum(pred),1);
        predictions(round(10.*epoch./iter),:) = pred./sum(pred);
    end
    
    p_y_x = F(Wy*A(z)).*TransitionFunction(z,0)./sum(F(Wy*A(z)).*TransitionFunction(z,0));
    pred = conv(fx_i,p_y_x,'same');
    Loss(epoch) = Cost(fy_j./sum(fy_j),pred./sum(pred),1);
    
    dLlx = (1/length(x_i(Rx))).*sum(f(Wx*A(x_i(Rx)))./F(Wx*A(x_i(Rx))).*A(x_i(Rx)),2);
    dLly = (1/length(y_j(Ry))).*sum(f(Wx*A(y_j(Ry)))./F(Wx*A(y_j(Ry))).*A(y_j(Ry)),2);
    
    Lrx = (1/length(x_i_(Rx_))).*sum(F(Wx*A(x_i_(Rx_)))./p_x_i_(Rx_),2);

    dLrx = (1/length(y_j_(Ry_))).*sum(F(Wy*A(y_j_(Ry_))),2).*(1/length(x_i_(Rx_))).*sum(f(Wx*A(x_i_(Rx_)))./p_x_i_(Rx_).*A(x_i_(Rx_)),2);
    dLry = (1/length(y_j_(Ry_))).*sum(f(Wy*A(y_j_(Ry_))).*A(y_j_(Ry_)),2).*Lrx;
    
    dLx = Dx.*transpose(dLlx-dLrx);
    dLy = Dy.*transpose(dLly-dLry);
    
    Gradx(:,epoch) = transpose(Dx).*(dLlx-dLrx);
    Grady(:,epoch) = transpose(Dy).*(dLly-dLry);
    
    if any(isnan(dLx))
        warning('Numerical Error')
        break
    elseif any(isinf(dLx))
        warning('Numerical Error')
        break
    end
    if any(isnan(dLy))
        warning('Numerical Error')
        break
    elseif any(isinf(dLy))
        warning('Numerical Error')
        break
    end
    
    [dWx,Ex,EDx] = AdaDelta(Ex,EDx,dLx);
    [dWy,Ey,EDy] = AdaDelta(Ey,EDy,dLy);
    
    Wx = Wx + dWx;
    Wy = Wy + dWy;
    
end

figure
plot(Loss,'k','linewidth',3);

p_y_x = F(Wy*A(z)).*(TransitionFunction(z,0))./sum(F(Wy*A(z)).*(TransitionFunction(z,0)));

pred = conv(fx_i,p_y_x,'same');

vals.Wx = Wx;
vals.Wy = Wy;

vals.PY = Wy*A(z);

vals.p_y_x = p_y_x;
vals.pred = pred;

vals.z = z;

vals.Loss = Loss;

vals.predictions = predictions;

vals.Gradx = Gradx;
vals.Grady = Grady;

end
