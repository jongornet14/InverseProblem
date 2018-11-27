function dL = Gradient(Bx,By,f,x_i,y_j,x_i_,z,num_F,pArray,k,grad,dim)

dP1dBx = zeros(dim,2);
dP1dBy = zeros(dim,2);
dP2dBx = zeros(dim,2);
dP2dBy = zeros(dim,2);

dL = zeros(dim,2);

f1 = f.f1;
f2 = f.f2;
F  = f.F;
S  = f.S;
s  = f.s;

Fy = zeros(dim,2);

if grad == 'x'
    
    for dd = 1:dim
    
    %                  %1/m                   %dphi/dalpha or dphi/dbeta
    %                  %
    dP1dBx(dd,1) = sum((1/length(x_i(dd,:))).*s(FunctionSummation(Bx,F,x_i(dd,:),num_F))./S(FunctionSummation(Bx,F,x_i(dd,:),num_F)).*f1(x_i(dd,:),k));
    dP1dBx(dd,2) = sum((1/length(x_i(dd,:))).*s(FunctionSummation(Bx,F,x_i(dd,:),num_F))./S(FunctionSummation(Bx,F,x_i(dd,:),num_F)).*f2(x_i(dd,:),k));
    
    for pX_i_ = 1:length(x_i_(dd,:))
        
        y_j_ = Sample(f.p_y_x(z,x_i_(dd,pX_i_)),z(dd,:),f.num_samples);
        
        Fy = (1/length(y_j_)).*sum(S(FunctionSummation(By,F,y_j_,num_F)));

        dP2dBx(dd,1) = sum((Fy./length(x_i_(dd,:))).*(s(FunctionSummation(Bx,F,x_i_(dd,:),num_F)).*f1(x_i_(dd,:),k))./pArray(dd,:));
        dP2dBx(dd,2) = sum((Fy./length(x_i_(dd,:))).*(s(FunctionSummation(Bx,F,x_i_(dd,:),num_F)).*f2(x_i_(dd,:),k))./pArray(dd,:));
    
    end
    
    end
    
    dL = dP1dBx - dP2dBx;
    
end

if grad == 'y'
    
    for dd = 1:dim

    dP1dBy(dd,1) = sum((1/length(y_j(dd,:))).*s(FunctionSummation(By,F,y_j(dd,:),num_F))./S(FunctionSummation(By,F,y_j(dd,:),num_F)).*f1(y_j(dd,:),k));
    dP1dBy(dd,2) = sum((1/length(y_j(dd,:))).*s(FunctionSummation(By,F,y_j(dd,:),num_F))./S(FunctionSummation(By,F,y_j(dd,:),num_F)).*f2(y_j(dd,:),k));
    
    for pX_i_ = 1:length(x_i_(dd,:))

        y_j_ = Sample(cumsum(f.p_y_x(z,x_i_(dd,pX_i_))./sum(f.p_y_x(z,x_i_(dd,pX_i_)))),z(dd,:),f.num_samples);
        
        Fy(1) = (1/length(y_j_)).*sum(s(FunctionSummation(By,F,y_j_,num_F)).*f1(y_j_,k));
        Fy(2) = (1/length(y_j_)).*sum(s(FunctionSummation(By,F,y_j_,num_F)).*f2(y_j_,k));

        dP2dBy(dd,1) = sum((Fy(1)./length(x_i_(dd,:))).*(S(FunctionSummation(Bx,F,x_i_(dd,:),num_F))./pArray(dd,:)));
        dP2dBy(dd,2) = sum((Fy(2)./length(x_i_(dd,:))).*(S(FunctionSummation(Bx,F,x_i_(dd,:),num_F))./pArray(dd,:)));
    
    end
    
    end
    
    dL = dP1dBy - dP2dBy;
    
end

end