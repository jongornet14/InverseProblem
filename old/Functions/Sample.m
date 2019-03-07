function [XY_ij,p] = Sample(fsample,z,num_samples)

XY_ij = zeros(length(z(:,1)),num_samples);
p = zeros(1,num_samples);

for ss = 1:num_samples
    
    P = zeros(1,ceil(length(z(1,:))/2));
    loc = zeros(length(z(:,1)),ceil(length(z(1,:))/2));
    
    for ij = 1:ceil(length(z(1,:))/10)
        
        xy = randi([1,length(z(1,:))],length(z(:,1)),1);
        
        if length(z(:,1)) == 1 
            P(ij) = fsample(xy);
            loc(1,ij) = z(1,xy);
        elseif length(z(:,1)) == 2
            P(ij) = fsample(xy(1),xy(2));
            loc(1,ij) = z(1,xy(1));
            loc(2,ij) = z(2,xy(2));
        elseif length(z(:,1)) == 3
            P(ij) = fsample(xy(1),xy(2),xy(3));
            loc(1,ij) = z(1,xy(1));
            loc(2,ij) = z(2,xy(2));
            loc(3,ij) = z(3,xy(3));
        end
        
    end
    
    id = find(P == max(P));
    XY_ij(:,ss) = loc(:,id(randi([1,length(id)],1)));
    p(ss) = P(id(randi([1,length(id)],1)))./sum(fsample(:));
    
end

end
        