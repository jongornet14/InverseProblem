function F = SampleIntegrateGradientP1(B,f,S,s,xy_ij,num_F,k)

F = 0;

% loop through each sample
for ij = 1:length(xy_ij(1,:))  
    F = F + (1/length(xy_ij(1,:))).*s(FunctionSummation(B,F,xy_ij(ij),num_F)./S(FunctionSummation(B,F,xy_ij(ij),num_F)).*f(xy_ij(ij),k);
end

end