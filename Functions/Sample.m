function [XY_ij,p] = Sample(fsample,z,num_samples)

XY_ij = zeros(length(z(:,1)),num_samples);
p = zeros(length(z(:,1)),num_samples);

for ss = 1:num_samples
    
    % make sure you are not getting samples of zero probability
    r = min(fsample) + (max(fsample) - min(fsample)).*rand(length(z(:,1)),1);
    
    for d = 1:length(z(:,1))
        z_loc = [];
        dz = z(1,2) - z(1,1);
        while true 
        for k = 1:length(z(1,:))
            if find(abs(fsample(d,k) - r(d)) < dz)
                z_loc = [z_loc k];
            end
        end
        if length(z_loc) > 0
            break
        else
            dz = dz + (z(1,2) - z(1,1));
        end
        end
        XY_ij(d,ss) = z(z_loc(randi(length(z_loc))));
        p(d,ss) = r(d);
    end
    
end

end
        