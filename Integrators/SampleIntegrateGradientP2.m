function Fx = SampleIntegrateGradientP2(Bx,By,f,S,s,x_i,y_j,num_F,pArray,k,type)

Fx = 0;
Fy = 0;
F = 0;

if type == 'x'
    
    % loop through each sample
    for j = 1:length(y_j(1,:))  
        Fy = Fy + (1/length(y_j(1,:))).*S(FunctionSummation(By,f,y_j(j),num_F));
    end
    for i = 1:length(x_i(1,:))  
        Fx = Fx + (Fy./length(x_i(1,:))).*(s(FunctionSummation(Bx,f,x_i(i),num_F)).*f(x_i(i),k))./pArray(i);
    end
    
end

if type == 'y'
    
    % loop through each sample
    for j = 1:length(y_j(1,:))  
        Fy = Fy + (1/length(y_j(1,:))).*s(FunctionSummation(By,f,y_j(j),num_F)).*f(y_j(j),k);
    end
    for i = 1:length(x_i(1,:))  
        Fx = Fx + (Fy./length(x_i(1,:))).*(S(FunctionSummation(Bx,f,x_i(i),num_F))./pArray(i));
    end
    
end

end