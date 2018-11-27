function C = Cost(Py_j,Py_j_,dim)

% C = 0;
% 
% for dd = 1:dim
%     for k = 1:length(Py_j(dd,:))
%         C = C + Py_j(dd,k).*log((Py_j(dd,k)+1e-5)/(Py_j_(dd,k)+1e-5));
%     end
% end

C = 0;
    
for k = 1:length(Py_j)
    C = C + sum(abs(Py_j(k)-Py_j_)).^2;
end
   
end