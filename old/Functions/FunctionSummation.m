function BkFk = FunctionSummation(B,F,xy_ij,num_F)

BkFk = 0;
    
for k = 1:num_F
    BkFk = BkFk + F(B(:,k,1),B(:,k,2),xy_ij,k);
end

end