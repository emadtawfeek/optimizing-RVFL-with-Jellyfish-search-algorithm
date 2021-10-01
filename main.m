

% Clear workspace
clc, clear, close;
% Prepare dataset
 

lb = [100, 0, 1, 0.0001];   % lower bound: [Number of neurons, Bias, Scalemode, Scale, Seed]
ub = [1000, 1, 3, 0.9999]; % upper bound
dim = 4;

% measure elasped time for jellyfish algorithm
% time_jellyfish = timeit(jellyfish(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim));
% fprintf('>>> time_jellyfish %f', time_jellyfish);

% time_pso = timeit(pso(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim));
% fprintf('>>> time_jellyfish %f', time_pso);
tic;

% select algorithm
algorithms = {'Jellyfish','GA',  'PSO', 'GWO' };
[sel_algo, Ok] = listdlg('PromptString', 'Select optimization algorithms.', ...
    'SelectionMode', 'single', ...
    'ListString', algorithms);

pause(0.03);

if Ok == 0
    return;
end

for  p=1:30
    
    fprintf('\n - iteration :  %d \n', p);
    x = xlsread('sorted.xlsx');
% load('sFeat.mat');  % Selected features from JSA-FS
% 
%x = readmatrix("feature_vectors_syscallsbinders_frequency_5_Cat.csv");
label = x(:, end);

% % histogram
% histogram(label)  % histcounts(label)

% x = readmatrix("upsampled_feat.csv");
% label = readmatrix("upsampled_label.csv");
% x = x(:, 2:end);
% label = label(:, 2: end);

% normalize data to [0, 1]
x = normalize(x,'range');
x=abs(x);
label = normalize(label,'range');
label=abs(label);

r = randi([50 100],1);
Numf = r;  % you can change this as you want
 fprintf('\n - r :  %d \n', r);
 x = fsrnca(x, label, Numf);
 

% Split data into train and test randomly
TOTAL_NUM_ROWS = length(label);
NUM_TRAIN = round(TOTAL_NUM_ROWS * 0.8);
% train_i = randi([1, TOTAL_NUM_ROWS], 1, NUM_TRAIN);
train_i = randperm(TOTAL_NUM_ROWS, NUM_TRAIN);
all_i = 1: TOTAL_NUM_ROWS;
test_i = setdiff(all_i, train_i);

train_X = x(train_i, :);
test_X = x(test_i, :);
train_y = label(train_i, :);
test_y = label(test_i, :);
% train_X = x(train_i, :);
% test_X = train_X;
% 
% train_y = label(train_i, :);
% test_y = train_y;

save('splited_data.mat', 'train_X', 'train_y', 'test_X', 'test_y');
% Load data
load('splited_data.mat');
% run the selected algorithm
switch sel_algo
    case 1,[u, fval, ~, ~] = jellyfish(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);
    case 2, [u, fval] = my_ga(lb, ub, dim);
    case 3, [best, ~] = pso(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim); fval = best.O; u = best.X;
    case 4, [fval, u, ~] = gwo(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);

%     case 2, [fval, u, ~] = da(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);
%     case 3, [fval, u, ~, ~, ~, ~] = goa(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);
%     case 5, [fval, u, ~] = woa(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);
%     case 7, [fval, u, ~] = ssa(train_X, train_y, test_X, test_y, @objFunc, lb, ub, dim);
   
  
end
diary on;
fprintf('\n - sel_algo numer:  %d \n ', sel_algo);
% Print the optimized parameters
fprintf('\n ==== Best Hyper Parameter ====');
fprintf('\n - Number of neurons:  %d', u(1));
fprintf('\n - Bias:  %d', u(2));
fprintf('\n - Scalemode:  %d', u(3));
fprintf('\n - Scale:  %.4f', u(4));
% fprintf('\n - Seed:  %d', 50);
fprintf('\n - Best mse:  %.4f \n', fval);

% optimized option
opt_option.N = u(1);          % Number of neurons
opt_option.bias = u(2);       % Bias
opt_option.Scalemode = u(3);  % Scale Mode
opt_option.Scale = u(4);      % Scale
% opt_option.seed = 50;       % Seed

% non-optimized option
nonOpt_option.N = 300;              % Number of neurons
nonOpt_option.bias = 0;            % Bias
nonOpt_option.Scalemode = 2;       % Scale Mode
nonOpt_option.Scale = 0.5;         % Scale
% nonOpt_option.seed = 50;           % Seed

fprintf('\n ==== Evaluation result for optimized hyperparameters ====');

opt_eval_result = RVFL_train_val(train_X, train_y, test_X, test_y, opt_option, 1);
TrainOut_Opt = opt_eval_result.TrainOut;
TestOut_Opt = opt_eval_result.TestOut;
print_result(opt_eval_result);
toc;
fprintf('\n ========================================================================\n');
fprintf('\n ========================================================================\n');
diary off;
% ss=timeit(opt_eval_result);
% fprintf('time it ---optimized operation %d',ss);
tic;
fprintf('\n ==== Evaluation result for non-optimized hyperparameters ====');
nonOpt_eval_result = RVFL_train_val(train_X, train_y, test_X, test_y, nonOpt_option, 1);
TrainOut_NonOpt = nonOpt_eval_result.TrainOut;
TestOut_NonOpt = nonOpt_eval_result.TestOut;
print_result(nonOpt_eval_result);
toc;

% sss=timeit(nonOpt_eval_result);
% fprintf('time it ---non - optimized operation %d',sss);
end
