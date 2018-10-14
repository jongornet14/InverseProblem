function C = Cost(Py_j,Py_j_,dim)

C = 0;

for dd = 1:dim
    for k = 1:length(Py_j(dd,:))
        if Py_j(dd,k) > 0 && Py_j_(dd,k) > 0
            C = C + Py_j(dd,k).*log(Py_j(dd,k)/Py_j_(dd,k));
        end
    end
end

end