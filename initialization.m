function pop=initialization(num_pop,nd,Ub,Lb)
% num_pop: Number of population;
% nd: Number of dimention; e.g: nd=4;
% Ub: Matrix of Upper bound,e.g:[1 1 1 1];
% Lb: Matrix of lower bound,e.g:[-1 0 -2 3];
for i = 1: num_pop
    for j = 1: nd
        if j == 4
            pop(i, j) = rand(1, 1);  % not integer for scale
        else
            pop(i, j) = randi([Lb(j), Ub(j)]); % integer for others
        end
    end
end
