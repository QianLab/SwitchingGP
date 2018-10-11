function negLogEmission = emissionMTGPFeat(signal, time, feature_model, gp_model, feature_set_index, feature_subsets)
%EMISSIONMTGP Summary of this function goes here
%   Detailed explanation goes here

% Constants
num_context = length(gp_model.coeffs);
rank_approx = gp_model.rank_approx;
num_features = rank_approx;

num_obs = ones(num_features, 1);

% Feature model
means = feature_model.means;
pca_coeffs = feature_model.pca_coeffs;

% GP model
gp_coeffs = gp_model.coeffs;
cov_func = gp_model.cov_func;
rank_approx = gp_model.rank_approx;

time = time(:);

negLogEmission = zeros(num_context, 1);
for j = 1 : num_context
  for f = feature_set_index
    % Feature Extraction
    mean_jf = means{j}{f};
    signal_features = (signal(:, feature_subsets{f}) - mean_jf) * pca_coeffs{j}{f};

    % Convert data format to fit with GP function
    task_index = kron(1 : size(signal_features, 2), ones(1, size(signal_features, 1)))';
    time_index = repmat(1 : size(signal_features, 1), 1, size(signal_features, 2))';

    signal_features = signal_features(:);

    log_theta = gp_coeffs{j}{f};

    negLogEmission(j) = negLogEmission(j) + nmargl_mtgp([], log_theta, cov_func, time, signal_features, num_features, rank_approx, num_obs, task_index, time_index, []);
  end
end

end
