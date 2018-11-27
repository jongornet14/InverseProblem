function C = Cost(Py_j,Py_j_,dim)

% C = 0;

% for dd = 1:dim
%     for k = 1:length(Py_j(dd,:))
%         C = C + Py_j(dd,k).*log((Py_j(dd,k)+1e-8)/(Py_j_(dd,k)+1e-8));
%     end
% end
  
C = -sum(Py_j.*log(Py_j_+1e-8));

%C = sum((Py_j-Py_j_).^2);
   
end