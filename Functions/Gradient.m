function dL = Gradient(Bx,By,f,x_i,y_j,x_i_,y_j_,num_F,pArray,k,grad,dim)

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
    %Log Part
    for i = 1:length(x_i(dd,:))
        fS = s(FunctionSummation(Bx,F,x_i(dd,i),num_F))./S(FunctionSummation(Bx,F,x_i(dd,i),num_F));
        dP1dBx(dd,1) = dP1dBx(dd,1) + (1/length(x_i(dd,:))).*fS(dd).*f1(x_i(dd,i),k);
        dP1dBx(dd,2) = dP1dBx(dd,2) + (1/length(x_i(dd,:))).*fS(dd).*f2(x_i(dd,i),k);
    end
    
    % loop through each sample
    for j = 1:length(y_j_(dd,:))  
        Fy = Fy + (1/length(y_j_(dd,:))).*S(FunctionSummation(By,F,y_j_(dd,j),num_F));
    end
    for i = 1:length(x_i_(dd,:))  
        fS = s(FunctionSummation(Bx,F,x_i_(dd,i),num_F));
        dP2dBx(dd,1) = dP2dBx(dd,1) + (Fy(dd,1)./length(x_i_(dd,:))).*(fS(dd).*f1(x_i_(dd,i),k))./pArray(dd,i);
        dP2dBx(dd,2) = dP2dBx(dd,2) + (Fy(dd,2)./length(x_i_(dd,:))).*(fS(dd).*f2(x_i_(dd,i),k))./pArray(dd,i);
    end
    end
    
    dL = dP1dBx - dP2dBx;
    
end

if grad == 'y'
    
    for dd = 1:dim
    %Log Part
    for j = 1:length(y_j(dd,:))
        fS = s(FunctionSummation(By,F,y_j(dd,j),num_F))./S(FunctionSummation(By,F,y_j(dd,j),num_F));
        dP1dBy(dd,1) = dP1dBy(dd,1) + (1/length(y_j(dd,:))).*fS(dd).*f1(y_j(dd,j),k);
        dP1dBy(dd,2) = dP1dBy(dd,2) + (1/length(y_j(dd,:))).*fS(dd).*f2(y_j(dd,j),k);
    end
    % loop through each sample
    for j = 1:length(y_j_(dd,:))  
        fS = s(FunctionSummation(By,F,y_j_(dd,j),num_F));
        Fy(dd,1) = Fy(dd,1) + (1/length(y_j_(dd,:))).*fS(dd).*f1(y_j_(dd,j),k);
        Fy(dd,2) = Fy(dd,2) + (1/length(y_j_(dd,:))).*fS(dd).*f2(y_j_(dd,j),k);
    end
    for i = 1:length(x_i_(dd,:))  
        fS = S(FunctionSummation(Bx,F,x_i_(dd,i),num_F));
        dP2dBy(dd,1) = dP2dBy(dd,1) + (Fy(dd,1)./length(x_i_(dd,:))).*(fS(dd)./pArray(dd,i));
        dP2dBy(dd,2) = dP2dBy(dd,2) + (Fy(dd,2)./length(x_i_(dd,:))).*(fS(dd)./pArray(dd,i));
    end
    end
    
    dL = dP1dBy - dP2dBy;
    
end

end