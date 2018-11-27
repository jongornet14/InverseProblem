function [vals] = InverseSolver(params)

repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

vecField = 3;
hidden_num = 12;

A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1));z.^transpose(0:vecField-1);exp(transpose(-1:1).*z)];
%A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1))];

F  = @(z) max(0,z)+1e-8;
f  = @(z) heaviside(z)+1e-8;

x_i = params.x_i;
y_j = params.y_j;

fx_i = params.fx_i;
fy_j = params.fy_j;

fF = params.fF;

p_x_i_ = zeros(1,length(x_i));
p_y_j_ = zeros(1,length(x_i));

x_i_ = zeros(1,length(x_i));
y_j_ = zeros(1,length(x_i));

z = linspace(0,10,100);
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

iter = 1e3;

Gradx = nan(hidden_num,iter);
Grady = nan(hidden_num,iter);

mapX = hist(x_i,z);
mapY = hist(y_j,z);
mapZ = hist(y_j_,z);

mapX = movmean(mapX./sum(mapX),10);
mapY = movmean(mapY./sum(mapY),10);
mapZ = movmean(mapZ./sum(mapZ),10);

figure
subplot(3,1,1)
plot(z,mapX)
subplot(3,1,2)
plot(z,mapY)
subplot(3,1,3)
plot(z,mapZ)

Loss = zeros(1,iter);

%%

if isfield(params,'Wx')
Wx = params.Wx;
else
Wx = rand(1,hidden_num); %x values
end
if isfield(params,'Wy')
Wy = params.Wy;
else
Wy = rand(1,hidden_num); %y values
end

Ex = zeros(1,hidden_num);
Ey = zeros(1,hidden_num);

EDx = zeros(1,hidden_num);
EDy = zeros(1,hidden_num);

%%

disp('Starting Training...');

for epoch = 1:iter
    
%     if rand > 0.8
    dropout = rand(1,hidden_num) <= 0.95;
%     else
%     dropout = rand(1,hidden_num) < 1;
%     end
    
    if mod(epoch,iter/10) == 0
        p_y_x = F(Wy*A(z)).*(TransitionFunction(z,0))./sum(F(Wy*A(z)).*(TransitionFunction(z,0)));
        pred = conv(mapX,p_y_x);
        pred = pred(1:length(z));
        disp(['Loss: ' num2str(Cost((fy_j-fF)./sum(fy_j-fF),pred-z.*max(pred)./10,1))]);
        disp([num2str(round(100.*epoch./iter)),'% Done!'])
        Loss(epoch) = Cost((fy_j-fF)./sum(fy_j-fF),pred-z.*max(pred)./10,1);
    end
    
    p_y_x = F(Wy*A(z)).*TransitionFunction(z,0)./sum(F(Wy*A(z)).*TransitionFunction(z,0));
    pred  = conv(mapX,p_y_x);
    pred = pred(1:length(z));
    Loss(epoch) = Cost((fy_j-fF)./sum(fy_j-fF),pred-z.*max(pred)./10,1);
    
    dLlx = (1/length(x_i)).*sum(f((dropout.*Wx)*A(x_i))./F((dropout.*Wx)*A(x_i)).*A(x_i),2);
    dLly = (1/length(y_j)).*sum(f((dropout.*Wx)*A(y_j))./F((dropout.*Wx)*A(y_j)).*A(y_j),2);
    
    Lrx = (1/length(x_i_)).*sum(F((dropout.*Wx)*A(x_i_))./p_x_i_,2);

    dLrx = (1/length(y_j_)).*sum(F((dropout.*Wy)*A(y_j_)),2).*(1/length(x_i_)).*sum(f((dropout.*Wx)*A(x_i_))./p_x_i_.*A(x_i_),2);
    dLry = (1/length(y_j_)).*sum(f((dropout.*Wy)*A(y_j_)).*A(y_j_),2).*Lrx;
    
    dLx = transpose(dLlx-dLrx);
    dLy = transpose(dLly-dLry);
    
    Gradx(:,epoch) = dLlx-dLrx;
    Grady(:,epoch) = dLly-dLry;
    
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

pred = conv(mapX,p_y_x);
pred = pred(1:length(z));

vals.Wx = Wx;
vals.Wy = Wy;

vals.PY = F(Wy*A(z));

vals.p_y_x = p_y_x;
vals.pred = pred;
vals.mapX = mapX;
vals.mapY = mapY;
vals.mapZ = mapZ;

vals.z = z;

vals.Loss = Loss;

end