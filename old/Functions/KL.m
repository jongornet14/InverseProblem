function L = KL(f1,f2,dz)

L = 0;

for k = 1:length(f1)
    
    L = L + f1(k).*log(f2(k)./f1(k)).*dz;
    
end

end
