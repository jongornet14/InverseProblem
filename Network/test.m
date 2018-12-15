z = -10:0.01:10;
pert = normpdf(z,0,1);
signal = lognpdf(z,1,1);
%signal = normpdf(z,0,3);
%signal = normpdf(z,0,1);

params.fx_i = pert./sum(pert);
params.fy_j = signal./sum(signal);

K = conv(TransitionFunction(z,0),pert,'same');
K = K./sum(K);

x_i = NyquistSample(z,pert);
y_j = NyquistSample(z,signal);

Rx_i = z(find(pert./sum(pert)==0));
Ry_j = z(find(signal./sum(signal)==0));

%%
Z = transpose(z);

dz = mean(diff(z));

r0 = sum(exp(-(Z-x_i).^2),2);
r0 = r0./sum(r0);

[x_i_,p_x_i_] = NyquistSample(z,r0);
    
%%
R = randi(length(x_i_),[1 1e4]);
x_i_ = x_i_(R);
p_x_i_ = p_x_i_(R);

y_j_ = zeros(1,1e3,length(x_i_));   

tic
for ii = 1:length(x_i_)
    
    P_Y_X = NyquistSample(Z,TransitionFunction(z,x_i_(ii)));
    y_j_(1,:,ii) = P_Y_X(randi(length(P_Y_X),[1,1e3]));
    
end
toc

%%

SY_j = zeros(1,length(x_i_));

tic
for ii = 1:length(x_i_)
    
    SY_j = lognpdf(x_i_(ii),1,1);
    
end
toc

%%
%Y = reshape(y_j_,[1e7,1]);

bins = linspace(z(1),z(end),1e2);
mapX = hist(x_i_,bins);
mapY = hist(y_j,bins);
mapZ = hist(SY_j,bins);

mapX = mapX./sum(mapX);
mapY = mapY./sum(mapY);
mapZ = mapZ./sum(mapZ);

figure
subplot(3,1,1)
plot(bins,mapX)
subplot(3,1,2)
plot(bins,mapY)
subplot(3,1,3)
plot(bins,mapZ)

%%

A = @(x) [ ones(size(x)); 2.*x; 4.*x.^2 - 2; 8.*x.^3 - 12.*x; 16.*x.^4 - 48.*x.^2 + 12; 32.*x.^5 - 160.*x.^3 + 120.*x; 64.*x.^6 - 480.*x.^4 + 720.*x.^2 - 120; 128.*x.^7 - 1344.*x.^5 + 3360.*x.^3 - 1680.*x ];
%A = @(x) [ x.^(transpose(1:10)).*exp(-x.^2);x.^(transpose(1:10)).*exp(-x.^2./2);x.^(transpose(1:10)).*exp(-x.^2./3) ];

params.x_i = x_i;
params.y_j = y_j;

params.x_i_ = x_i_;
params.y_j_ = y_j_;

params.X_I = A(x_i);
params.Y_J = A(y_j);

params.X_I_ = A(x_i_);
params.Y_J_ = A(y_j_);

params.NullX_I = A(Rx_i);
params.NullY_J = A(Ry_j);

params.p_x_i_ = p_x_i_;

%%

hidden_num = 30;

%%
m = length(x_i);
n = length(y_j);
m_ = 1e4;
n_ = 1e3;

X_I = params.X_I;
Y_J = params.Y_J;
X_I_ = params.X_I_;
Y_J_ = params.Y_J_;

X_i = X_I;
Y_j = Y_J;
X_i_ = X_I_;
Y_j_ = Y_J_;

P = params.p_x_i_;

% W = vals.W;

F = @(x) sqrt(1+x.^2)-1+1e-8;
f = @(x) x.*(1+x.^2).^(-1/2);

%%
L = - (1/m).*sum(log(F(W(1:hidden_num)*X_i)),2) - (1/n).*sum(log(F(W(hidden_num+1:2*hidden_num)*Y_j)),2) + 2.*(1/m_).*(1/n_).*sum((reshape(sum(F(sum(transpose(W(hidden_num+1:2*hidden_num)).*Y_j_,1)),2),[1,m_]).*F(W(1:hidden_num)*X_i_)./P)./(Null+1e-8),2)

%%

hidden_num = 8;

params.iter = 1e4;
params.W = W;
params.l = 1e-3.*ones(1,2*hidden_num);

params.path = '/Users/jonathangornet/Documents/GitHub/InverseSolver/Network/';

%%

[vals] = InverseSolver(params);

%%

pred = conv(pert,F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))),'same');

figure
subplot(3,1,1)
plot(z,params.fy_j,'k');
hold on
plot(z,K./sum(K),'b')
legend({'$$\{y_j\}$$ Distribution','Initial $$\{y_j\}$$ Predication'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Initiatial Prediction','Interpreter','latex','fontsize',20)
subplot(3,1,2)
plot(z,params.fy_j,'k');
hold on
plot(z,pred./sum(pred),'r');
legend({'$$\{y_j\}$$ Distribution','Adjusted $$\{y_j\}$$ Prediction'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Trained Prediction','Interpreter','latex','fontsize',20)
subplot(3,1,3)
plot(z,K./sum(K),'b')
hold on
plot(z,pred./sum(pred),'r');
legend({'Initial $$\{y_j\}$$ Distribution','Adjusted $$\{y_j\}$$ Prediction'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Trained Prediction','Interpreter','latex','fontsize',20)
%NiceSave('GaussianFig','~/Desktop/Figures',[])
%NiceSave('GaussianFigWithDropout','~/Desktop/Figures',[])
%NiceSave('IdentityFig','~/Desktop/Figures',[])
%NiceSave('LognormalFig','~/Desktop',[])
%NiceSave('GaussianFigWithDropout2','~/Desktop/Figures',[])

%%

figure
plot(1:params.iter,vals.Likelihood,'k','linewidth',2)
xlabel('Epoch','fontsize',20);ylabel('Likelihood','fontsize',20);title('Likelihood','fontsize',20)
xlim([9800 1e4])
%NiceSave('LognormalLoss','~/Desktop',[])
%NiceSave('GaussianLoss','~/Desktop/Figures',[])
%NiceSave('GaussianLossWithDropout','~/Desktop/Figures',[])
%NiceSave('IdentityLoss','~/Desktop/Figures',[])
%NiceSave('GaussianLossWithDropout2','~/Desktop/Figures',[])

%%

sum(z.^2.*vals.pred./sum(vals.pred))-sum(z.*vals.pred./sum(vals.pred)).^2
sum(z.^2.*K./sum(K))-sum(z.*K./sum(K)).^2
sum(z.^2.*signal./sum(signal))-sum(z.*signal./sum(signal)).^2

%%


figure;plot(z,F(W(1:hidden_num)*A(z)))

figure
plot(z,(conv(pert,F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))),'same')+1e-8)./(params.fy_j./sum(params.fy_j)+1e-8))

%%
figure
plot(z,(conv(pert,F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))),'same')+1e-8))

%%
figure
plot(z,F(W(hidden_num+1:2*hidden_num)*A(z)))

%%
figure
plot(F(W(hidden_num+1:2*hidden_num)*NullY_j))