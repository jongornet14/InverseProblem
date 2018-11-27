z = -10:0.01:10;
pert = normpdf(z,0,1);
%signal = normpdf(z,0,3);
%signal = lognpdf(z,1,1);
signal = normpdf(z,0,3);
%signal = normpdf(z,0,1);

id = find(isnan(signal));
signal(id) = 1;

params.fx_i = pert./sum(pert);
params.fy_j = signal./sum(signal);

K = conv(TransitionFunction(z,0),pert,'same');
K = K./sum(K);

% figure
% subplot(3,1,1)
% plot(z,params.fx_i);
% subplot(3,1,2)
% plot(z,params.fy_j)
% subplot(3,1,3)
% plot(z,K)

%pert = (pert-min(pert))./(max(pert)-min(pert));
signal = (signal-min(signal))./(max(signal)-min(signal));

x_i = NyquistSample(z,pert);
y_j = NyquistSample(z,signal);

% x_i = [x_i (10+10).*rand(1,1e3)-10];
% y_j = [y_j (10+10).*rand(1,1e3)-10];

%%
Z = transpose(z);

dz = mean(diff(z));

r0 = sum(exp(-(Z-x_i).^2),2);
r0 = r0./sum(r0);

for ii = 1:length(x_i)
    
    r = rand;
    
    [c,index] = min(abs(r-cumsum(r0)));
    x_i_(ii) = z(index);
    p_x_i_(ii) = r;
    
end

P_Y_X = conv(TransitionFunction(z,0),r0,'same');
P_Y_X = P_Y_X./sum(P_Y_X);

Y = NyquistSample(Z,P_Y_X);

R = randi([1 length(Y)],[1 1e5]);
y_j_ = Y(R);

bins = linspace(z(1),z(end),1e2);
mapX = hist(x_i,bins);
mapY = hist(y_j,bins);
mapZ = hist(Y,bins);

mapX = mapX./sum(mapX);
mapY = mapY./sum(mapY);
mapZ = mapZ./sum(mapZ);

%%

figure
subplot(3,1,1)
plot(bins,mapX)
subplot(3,1,2)
plot(bins,mapY)
subplot(3,1,3)
plot(bins,mapZ)

%%

params.x_i = x_i;
params.y_j = y_j;
params.iter = 1e4;

params.Wx = vals.Wx;
params.Wy = vals.Wy;

[vals] = InverseSolver(params);

%%

figure
subplot(3,1,1)
plot(z,params.fy_j./sum(params.fy_j),'k');
hold on
plot(z,K./sum(K),'b')
legend({'$$\{y_j\}$$ Distribution','Initial $$\{y_j\}$$ Predication'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Initiatial Prediction','Interpreter','latex','fontsize',20)
subplot(3,1,2)
plot(z,params.fy_j./sum(params.fy_j),'k');
hold on
plot(vals.z,vals.pred./sum(vals.pred),'r');
legend({'$$\{y_j\}$$ Distribution','Adjusted $$\{y_j\}$$ Prediction'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Trained Prediction','Interpreter','latex','fontsize',20)
subplot(3,1,3)
plot(z,K./sum(K),'b')
hold on
plot(vals.z,vals.pred./sum(vals.pred),'r');
legend({'Initial $$\{y_j\}$$ Distribution','Adjusted $$\{y_j\}$$ Prediction'},'Interpreter','latex')
xlabel('$$x$$ Space','Interpreter','latex','fontsize',20);ylabel('$$p(\{y_j\})$$','Interpreter','latex','fontsize',20);title('Trained Prediction','Interpreter','latex','fontsize',20)
%NiceSave('GaussianFig','~/Desktop/Figures',[])
%NiceSave('GaussianFigWithDropout','~/Desktop/Figures',[])
%NiceSave('IdentityFig','~/Desktop/Figures',[])
%NiceSave('LognormalFig','~/Desktop',[])
%NiceSave('GaussianFigWithDropout2','~/Desktop/Figures',[])

%%

figure
plot(vals.Loss,'k','linewidth',2)
xlabel('Epoch','fontsize',20);ylabel('Loss','fontsize',20);title('Cross Entropy Loss','fontsize',20)
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
