lsfunction Test_9_17_2018(ii)

% Inverse Tests
% Jonathan Gornet
% Date: 9-17-2018

repopath = '/scratch/jmg1030/InverseProblems/InverseSolver';
addpath(genpath(repopath));

%% N(0,1) to N(0,1)

if ii == 1
randX = randn(1,1e5);
randY = randn(1,1e5);

FourierSolver(randX,randY,'9-18-2017','N1-N1')
end

%% N(0,1) to N(0,10)

if ii == 2
randX = randn(1,1e5);
randY = normrnd(0,10,[1,1e5]);

FourierSolver(randX,randY,'9-18-2017','N1-H')
end

%% N(0,1) to L(0,1)

if ii == 3
randX = randn(1,1e5);
randY = lognrnd(0,1,[1,1e5]);

FourierSolver(randX,randY,'9-18-2017','N1-LN')

end

end