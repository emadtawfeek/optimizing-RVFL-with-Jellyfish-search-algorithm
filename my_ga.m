
function [x, fval] = my_ga(lb, Ub, dim)
disp('Running GA')
tic;
 dim=4;
%Ub(5)=50;
% You can change parameters of GA as you want.
 my_options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, 'MaxGenerations', 100, 'FitnessLimit', 0.016 );
[xx, fval] = ga(@myobjectfunc, dim, [], [], [], [], lb, Ub, [], my_options);
x = round(xx);
toc;
end
