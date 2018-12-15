function L = SchrodingerBridge(W)

z = -10:0.01:10;
pert = normpdf(z,0,1);
%signal = normpdf(z,0,3);
signal = lognpdf(z,1,1);
%signal = normpdf(z,0,3);
%signal = normpdf(z,0,1);

pert = (pert-min(pert))./(max(pert)-min(pert));
signal = (signal-min(signal))./(max(signal)-min(signal));

%%

Samples = 1e3;

x_i = NyquistSample(z,pert);
y_j = NyquistSample(z,signal);

Rx = randi(length(x_i),[1 Samples]);
Ry = randi(length(y_j),[1 Samples]);

x_i = x_i(Rx);
y_j = y_j(Ry);

r0 = sum(exp(-(transpose(z)-x_i).^2),2);
r0 = r0./sum(r0);

[x_i_,p_x_i_] = NyquistSample(z,signal);

Rx_ = randi(length(x_i_),[1 Samples]);

x_i_ = x_i_(Rx_);
p_x_i_ = p_x_i_(Rx_);

y_j_ = zeros(1,Samples,Samples);   

for ii = 1:Samples
    
    P_Y_X = NyquistSample(transpose(z),TransitionFunction(z,x_i_(ii)));
    y_j_(1,:,ii) = P_Y_X(randi(length(P_Y_X),[1,Samples]));
    
end

A = @(x) [ ones(size(x)); 2.*x; 4.*x.^2 - 2; 8.*x.^3 - 12.*x; 16.*x.^4 - 48.*x.^2 + 12; 32.*x.^5 - 160.*x.^3 + 120.*x; 64.*x.^6 - 480.*x.^4 + 720.*x.^2 - 120; 128.*x.^7 - 1344.*x.^5 + 3360.*x.^3 - 1680.*x ];
hidden_num = 8;

X_i = A(x_i);
Y_j = A(y_j);

X_i_ = A(x_i_);
Y_j_ = A(y_j_);

P = p_x_i_;

m = Samples;
n = Samples;
m_ = Samples;
n_ = Samples;

F = @(x) sqrt(1+x.^2)-1;
%f = @(x) x.*(1+x.^2).^(-1/2);

L = - (1/m).*sum(log(F(W(1:hidden_num)*X_i)),2) - (1/n).*sum(log(F(W(hidden_num+1:2*hidden_num)*Y_j)),2) + 2.*(1/m_).*(1/n_).*sum(reshape(sum(F(sum(transpose(W(hidden_num+1:2*hidden_num)).*Y_j_,1)),2),[1,m_]).*F(W(1:hidden_num)*X_i_)./P,2);

if isinf(L)
    L = 1e50;
end

end
