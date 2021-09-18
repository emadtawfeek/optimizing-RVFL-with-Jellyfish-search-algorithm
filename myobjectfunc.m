function [mse] = myobjectfunc(x)
    load('splited_data.mat');
    mse = objFunc(train_X, train_y, test_X, test_y, x);
%    disp(mse)
end