function C = Cost(Py_j,Py_j_,dim)
  
C = sum(Py_j.*log((Py_j+1e-8)./(Py_j_+1e-8)));
   
end