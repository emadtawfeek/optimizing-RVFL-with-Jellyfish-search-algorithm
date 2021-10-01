function selectedf = fsrnca(X, y, numf)
    % feature selction
    md1 = fsrnca(X, y);
    weights = md1.FeatureWeights;
    % ordered weights
    [W, W_order] = sort(weights, 'descend');
    % choose most importand 100 features only
    selected_indices = W_order(1:num_feats);
    selectedf = X(:, selected_indices);
end
