function [signal_preds, variance_preds] = predictNextStep(signal, time, t, feature_model, gp_model, sub_num)

% Feature model
pca_coeffs = feature_model.pca_coeffs;
means = feature_model.means;
num_features = size(pca_coeffs{1},2);
num_contexts = length(pca_coeffs);
num_obs = ones(num_features, 1);

% GP Model
rank_approx = gp_model.rank_approx;
num_features = rank_approx;

time_obs = time(1:t-1);
time_obs = time_obs(:);
signal_obs = signal(time_obs, :);

signal_preds = zeros(num_contexts, num_features);
variance_preds = zeros(num_contexts, num_features);
for j = 1 : num_contexts
  % Feature Extraction
  mean_j = means{j}{sub_num};
  signal_features = (signal_obs - mean_j) * pca_coeffs{j};

  % Convert data format to fit with GP function
  task_index = kron(1 : size(signal_features, 2), ones(1, size(signal_features, 1)))';
  time_index = repmat(1 : size(signal_features, 1), 1, size(signal_features, 2))';

  signal_obs_in = signal_features(:);
  time_pred = time(t);
  
  data_obs = {gp_model.cov_func, time_obs, signal_obs_in, num_features, rank_approx, num_obs, task_index, time_index};

  [signal_pred, Vpred] = predict_popmtgp_all_tasks(gp_model.coeffs{j}, data_obs, time_pred);
  signal_preds(j, :) = signal_pred;
  variance_preds(j, :) = Vpred;
end

  
end
