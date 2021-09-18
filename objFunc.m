function mse_val = objFunc(train_X, train_y, test_X, test_y, x)
    option.N = round(x(1));         % Number of neurons
    option.bias = round(x(2));      % Bias: 0, 1
    option.Scalemode = round(x(3)); % Scale mode: 0, 1, 2
    option.Scale = x(4);     % Scale: 0 ~ 1
%     option.seed = round(x(5));      % Seed: 1 ~ 50option

%      option.N = 300;        % Number of neurons
%     option.bias = 1;      % Bias: 0, 1
%     option.Scalemode = 2; % Scale mode: 0, 1, 2
%     option.Scale = 0.5;     % Scale: 0 ~ 1
% %     option.seed = 50;      % Seed: 1 ~ 50
    eval_result = RVFL_train_val(train_X, train_y, test_X, test_y, option);
    % mse_val = eval_result.test.mse;
    mse_val = 1 - eval_result.test.acc;
end