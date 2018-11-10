function [z] = Convolution(x,y,space)

dspace = mean(diff(space));

z = zeros(1,length(space));

for t1 = 1:length(space)
    for t2 = 1:length(space)
        
        z(t1) = x(t1).*y(t2);
        
    end
end

end