function [vals] = InverseSolver(params)

repopath = '~/Documents/Github/InverseSolver';
addpath(genpath(repopath));

hidden_num = 8;

%A = @(x) [ x.^(transpose(1:10)).*exp(-x.^2);x.^(transpose(1:10)).*exp(-x.^2./2);x.^(transpose(1:10)).*exp(-x.^2./3) ];
A = @(x) [ ones(size(x)); 2.*x; 4.*x.^2 - 2; 8.*x.^3 - 12.*x; 16.*x.^4 - 48.*x.^2 + 12; 32.*x.^5 - 160.*x.^3 + 120.*x; 64.*x.^6 - 480.*x.^4 + 720.*x.^2 - 120; 128.*x.^7 - 1344.*x.^5 + 3360.*x.^3 - 1680.*x ];
%A = @(x) [ ones(size(x)); 2.*x; 4.*x.^2 - 2; 8.*x.^3 - 12.*x; 16.*x.^4 - 48.*x.^2 + 12; 32.*x.^5 - 160.*x.^3 + 120.*x; 64.*x.^6 - 480.*x.^4 + 720.*x.^2 - 120; 128.*x.^7 - 1344.*x.^5 + 3360.*x.^3 - 1680.*x; 256.*x.^8 - 3584.*x.^6 + 13440.*x.^4 - 13440.*x.^2 + 1680; 512.*x.^9 - 9216.*x.^7 + 48384.*x.^5 - 80640.*x.^3 + 30240.*x; 1024.*x.^10 - 23040.*x.^8 + 161280.*x.^6 - 403200.*x.^4 + 302400.*x.^2 - 30240];

F = @(x) sqrt(1+x.^2)-1+1e-8;
f = @(x) x.*(1+x.^2).^(-1/2)+1e-8;

x_i = params.x_i;
y_j = params.y_j;

fx_i = params.fx_i;
fy_j = params.fy_j;

z = -10:0.01:10;

%%

m = length(x_i);
n = length(y_j);
m_ = 1e4;
n_ = 1e3;

X_i = params.X_I;
Y_j = params.Y_J;
X_i_ = params.X_I_;
Y_j_ = params.Y_J_;

P = params.p_x_i_;

Null = sum(exp(-(params.x_i_ - transpose(params.y_j)).^2),1)./sum(sum(exp(-(params.x_i_ - transpose(params.y_j)).^2),1));

%Likelihood = zeros(1,params.iter);

%%
if isfield(params,'W')    
W = params.W;
else    
W = randn(1,2*hidden_num);
end

E = zeros(1,2*hidden_num);
ED = zeros(1,2*hidden_num);

GradW = zeros(2*hidden_num,params.iter);

l = params.l;

Lnum = 1;

%%

disp('Starting Training...');

for epoch = 1:params.iter
    
    dW = [- (1/m).*sum(f(W(1:hidden_num)*X_i)./F(W(1:hidden_num)*X_i).*X_i,2) + (1/m_).*(1/n_).*sum((reshape(sum(F(sum(transpose(W(hidden_num+1:2*hidden_num)).*Y_j_,1)),2),[1,m_]).*f(W(1:hidden_num)*X_i_)./P.*X_i_)./(Null+1e-8),2); 
          - (1/n).*sum(f(W(hidden_num+1:2*hidden_num)*Y_j)./F(W(hidden_num+1:2*hidden_num)*Y_j).*Y_j,2) + (1/m_).*(1/n_).*sum((reshape(sum(f(sum(transpose(W(hidden_num+1:2*hidden_num)).*Y_j_,1)).*Y_j_,2),[hidden_num,m_]).*F(W(1:hidden_num)*X_i_)./P)./(Null+1e-8),2)];
    
%     dW = [- sum((f(W(1:hidden_num)*A(z))./F(W(1:hidden_num)*A(z))).*A(z).*mean(diff(z)),2) + sum((sum(TransitionFunction(z,0).*f(W(1:hidden_num)*A(z)).*A(z).*mean(diff(z)),2).*F(W(hidden_num+1:2*hidden_num)*A(z)).*mean(diff(z)))./(fy_j+1e-8),2);
%           - sum((f(W(hidden_num+1:2*hidden_num)*A(z))./F(W(hidden_num+1:2*hidden_num)*A(z))).*A(z).*mean(diff(z)),2) + sum((sum(TransitionFunction(z,0).*F(W(1:hidden_num)*A(z)).*mean(diff(z)),2).*f(W(hidden_num+1:2*hidden_num)*A(z)).*A(z).*mean(diff(z)))./(fy_j+1e-8),2)];
%     
%     Likelihood(epoch) = - sum(log(F(W(1:hidden_num)*A(z))).*mean(diff(z))) - sum(log(F(W(hidden_num+1:2*hidden_num)*A(z))).*mean(diff(z))) + 2.*sum((sum(TransitionFunction(z,0).*F(W(1:hidden_num)*A(z)).*mean(diff(z))).*F(W(hidden_num+1:2*hidden_num)*A(z)).*mean(diff(z))+1e-8)./(fy_j+1e-8));

%     dW = [- sum((f(W(1:hidden_num)*A(z))./F(W(1:hidden_num)*A(z))).*A(z).*mean(diff(z)),2) + sum((sum(TransitionFunction(z,0).*f(W(1:hidden_num)*A(z)).*A(z).*mean(diff(z)),2).*F(W(hidden_num+1:2*hidden_num)*A(z)).*mean(diff(z))),2);
%           - sum((f(W(hidden_num+1:2*hidden_num)*A(z))./F(W(hidden_num+1:2*hidden_num)*A(z))).*A(z).*mean(diff(z)),2) + sum((sum(TransitionFunction(z,0).*F(W(1:hidden_num)*A(z)).*mean(diff(z)),2).*f(W(hidden_num+1:2*hidden_num)*A(z)).*A(z).*mean(diff(z))),2)];
%     
%     Likelihood(epoch) = - sum(log(F(W(1:hidden_num)*A(z))).*mean(diff(z))) - sum(log(F(W(hidden_num+1:2*hidden_num)*A(z))).*mean(diff(z))) + 2.*sum((sum(TransitionFunction(z,0).*F(W(1:hidden_num)*A(z)).*mean(diff(z))).*F(W(hidden_num+1:2*hidden_num)*A(z)).*mean(diff(z))+1e-8));
    
    if mod(epoch,25) == 0
        Likelihood(Lnum) = - (1/m).*sum(log(F(W(1:hidden_num)*X_i)),2) - (1/n).*sum(log(F(W(hidden_num+1:2*hidden_num)*Y_j)),2) + 2.*(1/m_).*(1/n_).*sum((reshape(sum(F(sum(transpose(W(hidden_num+1:2*hidden_num)).*Y_j_,1)),2),[1,m_]).*F(W(1:hidden_num)*X_i_)./P)./(Null+1e-8),2);
        disp(['Likelihood: ' num2str(Likelihood(Lnum))])
        pred = conv(fx_i,F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))),'same');
        pred = pred./sum(pred);
        disp(['Cost: ' num2str(Cost(pred,fy_j,1))])
        Lnum = Lnum + 1;
    end
    
    if mod(epoch,100) == 0
        
        p_y_x = F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0)));
        
        vals.W = W;
        vals.GradW = GradW;

        vals.l = l;

        vals.p_y_x = p_y_x;
        vals.pred = pred;

        vals.z = z;
        
        save([params.path 'NetworkWeightsTest'],'-struct','vals','-v7.3');
        
    end
    
    if Lnum > 1
        if Likelihood(Lnum) > Likelihood(Lnum - 1)
            l = l+1;
        end
    end
        
    if isnan(dW)
        break
    end
    if isinf(dW)
        break
    end
    
    GradW(:,epoch) = dW;
    
    %[dW,E,ED] = AdaDelta(E,ED,transpose(dW),l);
    
    %W = W - dW;
    W = W - transpose(10.^(-l-ceil(log10(abs(dW)))).*dW);
    
end

%%
p_y_x = F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0))./sum(F(W(hidden_num+1:2*hidden_num)*A(z)).*(TransitionFunction(z,0)));

pred = conv(fx_i,p_y_x,'same');

vals.W = W;
vals.GradW = GradW;

vals.l = l;

vals.p_y_x = p_y_x;
vals.pred = pred;

vals.z = z;

end
