repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

vecField = 5;
hidden_num = 17;

E = ones(hidden_num,1); %expander

%A  = @(z) z.^(transpose(0:hidden_num-1));
%A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1));z.^transpose(1:vecField);exp(transpose(-2:2).*z)];
A  = @(z) [cos(z.*(transpose(0:vecField-1)));sin(z.*transpose(0:vecField-1));exp(transpose(-3:3).*z)];

F  = @(z) max(0,z)+1e-8;
f  = @(z) heaviside(z)+1e-8;

s = 1;

x_i = randn(1,1e4);

%y_j = normrnd(0,3,[1,1e4]);
y_j = lognrnd(1,1,[1,1e4]);

x_i_ = x_i;%randn(1,1e4);

z = -25:0.01:25;

for ii = 1:length(x_i_)
    
    r = rand;
    [c,index] = min(abs(r-normpdf(z,x_i_(ii),1)));
    y_j_(ii) = z(index);
    p_x_i_(ii) = r;
    
end

m = 1;
n = 1e-1;
q = 0.9;

mx = 0;
vx = 0;
my = 0;
vy = 0;

iter = 1e3;

Gradx = nan(hidden_num,iter);
Grady = nan(hidden_num,iter);

bins = linspace(z(1),z(end),100);

mapX = hist(x_i,bins);
mapY = hist(y_j,bins);
mapZ = hist(y_j_,bins);

mapX = mapX./sum(mapX);
mapY = mapY./sum(mapY);
mapZ = mapZ./sum(mapZ);

figure
subplot(3,3,1)
plot(bins,mapX,'b','linewidth',3)
subplot(3,3,4)
plot(bins,mapY,'r','linewidth',3)
subplot(3,3,7)
plot(bins,mapZ,'k','linewidth',3)
hold on
plot(bins,mapX,'b','linewidth',3)

Loss = zeros(1,iter);

%%

Wx = rand(1,hidden_num); %x values
Wy = rand(1,hidden_num); %y values

dWx = zeros(1,hidden_num);
dWy = zeros(1,hidden_num);

Ex = zeros(1,hidden_num);
Ey = zeros(1,hidden_num);

EDx = zeros(1,hidden_num);
EDy = zeros(1,hidden_num);

%%
Ex = zeros(1,hidden_num);
Ey = zeros(1,hidden_num);

for epoch = 1:iter
    
    if mod(epoch,iter/10) == 0
        p_y_x = F(Wy*A(z)).*(normpdf(z,0,1)./10)./sum(F(Wy*A(z)).*(normpdf(z,0,1)./10));
        pred  = conv(normpdf(z,0,1)./10,p_y_x,'same');
        disp(['KL Loss: ' num2str(Cost(pred,lognpdf(z,1,1)./100,1))]);
        disp([num2str(round(100.*epoch./iter)),'% Done!'])
    end
    
    p_y_x = F(Wy*A(z)).*(normpdf(z,0,1)./10)./sum(F(Wy*A(z)).*(normpdf(z,0,1)./10));
    pred  = conv(normpdf(z,0,1)./10,p_y_x,'same');
    Loss(epoch) = Cost(pred,lognpdf(z,0,1)./100,1);
    
    Llx = (1/length(x_i)).*sum(log(F(Wx*A(x_i))),2);
    dLlx = (1/length(x_i)).*sum(f(Wx*A(x_i))./F(Wx*A(x_i)).*A(x_i),2);

    Lly = (1/length(y_j)).*sum(log(F(Wy*A(y_j))),2);
    dLly = (1/length(y_j)).*sum(f(Wy*A(y_j))./F(Wy*A(y_j)).*A(y_j),2);
    
    Lrx = (1/length(x_i_)).*sum(F(Wx*A(x_i_))./p_x_i_,2);
    Lry = (1/length(y_j_)).*sum(F(Wy*A(y_j_)),2).*Lrx;

    dLrx = (1/length(y_j_)).*sum(F(Wy*A(y_j_)),2).*(1/length(x_i_)).*sum(f(Wx*A(x_i_))./p_x_i_.*A(x_i_),2);
    dLry = (1/length(y_j_)).*sum(f(Wy*A(y_j_)).*A(y_j_),2).*Lrx;
    
    Lx = Llx - Lrx;
    Ly = Lly - Lry;
    
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
%     
%     [dWx,mx,vx] = Adam(mx,vx,dLx);
%     [dWy,my,vy] = Adam(my,vy,dLy);
    
    [dWx,Ex,EDx] = AdaDelta(Ex,EDx,dLx);
    [dWy,Ey,EDy] = AdaDelta(Ey,EDy,dLy);
    
%     dWx = (n./sqrt(Ex + 1e-8)).*dLx + randn(1,hidden_num);
%     dWy = (n./sqrt(Ey + 1e-8)).*dLy + randn(1,hidden_num);
       
    Wx = Wx + dWx;
    Wy = Wy + dWy;
    
end

figure
plot(Loss,'k','linewidth',3);

%%

figure
plot(z,F(Wy*A(z)))

p_y_x = F(Wy*A(z)).*(1./sqrt(2.*pi)).*exp(-z.^2./2)./sum(F(Wy*A(z)).*(1./sqrt(2.*pi).*exp(-z.^2./2)));

figure;plot(z,p_y_x)

figure
plot(z,conv(normpdf(z,0,1)./10,p_y_x,'same'))
% hold on
% plot(bins,mapY,'k','linewidth',3)
hold on
plot(z,lognpdf(z,1,1)./10)
%hold on
%plot(z,normpdf(z,0,3)./10)
hold on
plot(z,normpdf(z,0,1)./10)