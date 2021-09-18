% Prepare dataset
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

% feature selection
 Num_feats = 99;  % you can change this as you want
 x = fsrnca_feature_selection(x, label, Num_feats);

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
