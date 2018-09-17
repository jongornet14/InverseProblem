function F = SampleIntegrate(B,F,S,xy_ij,num_F,pArray)

L = zeros(length(xy_ij(:,1)),1);

% Integral for Importance Sampling
if length(pArray) > 1
    
    for dd = 1:length(xy_ij(:,1))
    % loop through each sample
    for ij = 1:length(xy_ij(1,:))  
        L(dd) = L(dd) + (1/length(xy_ij(dd,:))).*S(FunctionSummation(B,F,xy_ij(dd,ij),num_F))./pArray(dd,ij);
    end
    end

% Integral for Sampling
else
    
    for dd = 1:length(xy_ij(:,1))
    % loop through each sample
    for ij = 1:length(xy_ij(dd,:))  
        L(dd) = L(dd) + (1/length(xy_ij(dd,:))).*S(FunctionSummation(B,F,xy_ij(dd,ij),num_F));
    end
    end

end
end