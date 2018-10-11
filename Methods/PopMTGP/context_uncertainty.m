function [context_pred, alpha_prob, energy_costs, entropies, feature_sets] = context_uncertainty(signal, time, model, pred_lag, max_duration, sub_num, energy_cost)
%CONTEXT_UNCERTAINTY Summary of this function goes here
%   Detailed explanation goes here

import brml.*

% Feature model
feature_model = model.feature_model;
pca_coeffs = feature_model.pca_coeffs;
num_features = size(feature_model.pca_coeffs{1}, 2);

% HMM Model
p_trans = model.hmm_model;

% Gamma Process Model
gamma_model = model.gamma_model;

% GP Model
gp_model = model.gp_model;

% Constants
num_contexts = length(gamma_model);
time_length = length(time);
D = max_duration;

% Forward-Backward Prediction
alpha_prob = zeros(time_length, num_contexts);
log_alpha = zeros(time_length, num_contexts, max_duration);

context_pred = zeros(time_length, 1);
energy_costs = zeros(time_length, 1);
entropies = zeros(time_length, 1);

% Initial alpha and beta
feature_lengths = [];
feature_sets = zeros(time_length, num_features);
feature_sets(1, :) = 1;
num_samples = 10;
t = 1;
d = 1;

log_trans = sum(log(p_trans), 2);
log_alpha(t, :, d) = -emissionMTGP(signal(t, :), time(t), feature_model, gp_model, sub_num);

alpha_prob(t, :) = log_alpha(t, :, d);
[~, context_pred(t)] = max(log_alpha(t, :, d));

log_mix = log_trans + log_alpha(t, :, d);
log_mix = sum(log_mix, 2);

time_ind = ones(num_features, 1);
feat_ind = (1 : num_features)';

tic;

% Select best feature set
[mean_preds, variance_preds] = predictNextStep(signal, time, t+1, feature_model, gp_model, sub_num);
objective = Inf;
gm = gmdistribution(mean_preds, reshape(variance_preds', 1, size(variance_preds, 2), size(variance_preds, 1)), condexp(log_mix)');
for k = 4 : 3 : num_features
	time_ind_curr = [time_ind; 2 * ones(k, 1)];
    feature_lengths_curr = [feature_lengths, k];
	next_feature_sets = nchoosek(1 : num_features, k);
	for f = 1 : size(next_feature_sets, 1)
		feat_ind_curr = [feat_ind; next_feature_sets(f, :)'];
		log_sum_entropy = 0;
		for num = 1 : num_samples
			y_sample = random(gm);
			log_alpha_next = MTGPAlpha(log_alpha, t+1, signal(t, :), time, time_ind_curr, feat_ind_curr, max_duration, ...
								feature_model, gp_model, gamma_model, p_trans, sub_num, feature_lengths_curr, y_sample);
			alpha_next = condexp(log_alpha_next);
			alpha_next(log_alpha_next == 0) = [];
			alpha_next = alpha_next(:);

			log_alpha_next(log_alpha_next == 0) = [];
			log_alpha_next = log_alpha_next(:);

			log_sum_entropy = log_sum_entropy - sum(alpha_next .* log_alpha_next(:))/k;
		end

		log_sum_entropy = log_sum_entropy / num_samples;
		curr_value = log_sum_entropy + k * energy_cost * min(t, D);

		if curr_value < objective
			objective = curr_value;
			energy_costs(t) = k * energy_cost;
			entropies(t) = log_sum_entropy;
            feature_lengths_next = feature_lengths_curr;
			time_ind_next = time_ind_curr;
			feat_ind_next = feat_ind_curr;
			
			feature_sets(t+1, :) = zeros(1, num_features);
			feature_sets(t+1, next_feature_sets(f, :)) = 1;
		end
	end
end

disp(num2str(t));
toc;

% Recurse
for t = 2 : time_length
    tic;
	time_ind_curr = time_ind_next;
	feat_ind_curr = feat_ind_next;
    feature_lengths_curr = feature_lengths_next;

	[log_alpha_next, log_trans_next] = MTGPAlpha(log_alpha, t, signal(1 : t, :), time(1 : t), time_ind_curr, feat_ind_curr, ...
        max_duration, feature_model, gp_model, gamma_model, p_trans, sub_num, feature_lengths_curr, []);
	log_alpha(t, :, :) = log_alpha_next;

	alpha_prob(t, :) = sum(log_alpha(t, :, :), 3);
	[~, context_pred(t)] = max(sum(log_alpha(t, :, :), 3));

    log_mix = log_alpha_next;
	log_mix = sum(log_mix, 2);
    
    [mean_preds, variance_preds] = predictNextStep(signal, time, t+1, feature_model, gp_model, sub_num);
    objective = Inf;
    gm = gmdistribution(mean_preds, reshape(variance_preds', 1, size(variance_preds, 2), size(variance_preds, 1)), condexp(log_mix)');

	for k = 4 : 3 : num_features
		time_ind_curr = [time_ind_next; (t+1) * ones(k, 1)];
        feature_lengths_curr = [feature_lengths_next, k];
		next_feature_sets = nchoosek(1 : num_features, k);
		for f = 1 : size(next_feature_sets, 1)
			feat_ind_curr = [feat_ind_next; next_feature_sets(f, :)'];
			log_sum_entropy = 0;
			for num = 1 : num_samples
				y_sample = random(gm);
				log_alpha_next = MTGPAlpha(log_alpha, t+1, signal(t, :), time, time_ind_curr, feat_ind_curr, ...
										max_duration, feature_model, gp_model, gamma_model, p_trans, sub_num, feature_lengths_curr, y_sample);
				alpha_next = condexp(log_alpha_next);
				alpha_next(log_alpha_next == 0) = [];
				alpha_next = alpha_next(:);
				
				log_alpha_next(log_alpha_next == 0) = [];
				log_alpha_next = log_alpha_next(:);
				
				log_sum_entropy = log_sum_entropy - sum(alpha_next .* log_alpha_next(:))/k;
			end
	
			log_sum_entropy = log_sum_entropy / num_samples;
			curr_value = log_sum_entropy + k * energy_cost * min(t, D);
	
			if curr_value < objective
				objective = curr_value;
				energy_costs(t) = k * energy_cost;
				entropies(t) = log_sum_entropy;
                feature_lengths_next = feature_lengths_curr;
				time_ind_next = time_ind_curr;
				feat_ind_next = feat_ind_curr;
                
                feature_sets(t+1, :) = zeros(1, num_features);
                feature_sets(t+1, next_feature_sets(f, :)) = 1;
			end
		end
    end
    disp(num2str(t));
    toc;
end

disp('Done');


% for t = 2 : time_length
% 	% Forward step
% 	for d = 1 : D
% 		% Time Variables
% 		t_min_d_p1 = max(1, t-d+1);
% 		t_min_d = max(1, t-d);
% 		time_j = abs(time(t) - time(t_min_d_p1));
		
% 		% Emission
% 		log_emission = -emissionMTGP(signal(t_min_d_p1 : t, :), time(t_min_d_p1 : t), feature_model, gp_model, sub_num);
		
% 		for j = 1 : num_contexts
% 			% Sum variable
% 			log_sum = 0;
			
% 			% Gamma J
% 			gamma_j = gamma_model{j};
% 			log_gamma_j = log(gampdf(time_j, gamma_j(1), gamma_j(2)));
			
% % 			if log_gamma_j == -inf
% % 				log_gamma_j = 0;
% % 			end

% 			next_state = 1 : num_contexts;
% 			next_state(j) = [];
% 			for i = next_state
% 				% Gamma I
% 				gamma_i = gamma_model{i};
				
% 				for d_prime = 1 : D
% 					% Time variables
% 					t_min_d_dp_p1 = max(1, t - d - d_prime + 1);
% 					time_i = abs(time(t_min_d_dp_p1) - time(t_min_d));
					
% 					% Gamma I
% 					log_gamma_i = log(gampdf(time_i, gamma_i(1), gamma_i(2)));

% 					% Transition probability
% 					trans_ji = p_trans(j, i);
% 					log_trans = log(trans_ji) + log_gamma_i + log_gamma_j;
					
% 					if (t-d < 0)
% 						log_alpha_tilde = -Inf;
% 					else
% 						log_alpha_tilde = log_alpha(t_min_d, i, d_prime);
% 					end
					 
% 					log_sum = brml.logsumexp([log_sum; log_alpha_tilde + log_trans]);
% 				end
% 			end
% 			log_alpha(t, j, d) = log_emission(j) + log_sum;
% 		end
% 	end	
	
% 	% 	[~, context_pred(t)] = max(sum(log_alpha(t, :, :) + log_beta(t, :, :), 3));
% 	alpha_prob(t, :) = sum(log_alpha(t, :, :), 3);
% 	[~, context_pred(t)] = max(sum(log_alpha(t, :, :), 3));
% % 	disp(context_pred(t));
	
% % 	% Backward
% % 	t_next = min(t + pred_lag, length(time));
% % 	log_beta = zeros(length(time), num_contexts, max_duration);
% % 	for t_beta = t_next : -1 : t
% % 		
% % 		t_p1 = min(t_beta+1, t_next);
% % 		
% % 		% Observation probability
% % 		emissions = zeros(num_contexts, D);
% % 		for d_prime = 1 : D
% % 			t_plus_dp = min(t_beta+d_prime, t_next);
% % 			emissions(:, d_prime) = -emissionMTGP(signal(t_p1 : t_plus_dp, :), time(t_p1 : t_plus_dp), ...
% % 				feature_model, gp_model);
% % 		end
% % 		
% % 		% Transition probability
% % 		% Gamma J
% % 		gamma_js = zeros(num_contexts, D);
% % 		for j = 1 : num_contexts
% % 			gamma_j = gamma_model{j};
% % 			for d = 1 : D
% % 				t_min_d_p1 = max(1, t_beta-d+1);
% % 				time_j = abs(time(t_min_d_p1) - time(t));
% % 				log_gamma_j = log(gampdf(time_j, gamma_j(1), gamma_j(2)));
% % 				
% % 				gamma_js(j, d) = log_gamma_j;
% % 			end
% % 		end
% % 		
% % 		% Gamma I
% % 		gamma_is = zeros(num_contexts, D);
% % 		for i = 1 : num_contexts
% % 			gamma_i = gamma_model{i};
% % 			for d_prime = 1 : D
% % 				t_plus_dp = min(t_beta+d_prime, t_next);
% % 				time_i = abs(time(t_p1) - time(t_plus_dp));
% % 				log_gamma_i = log(gampdf(time_i, gamma_i(1), gamma_i(2)));
% % 				
% % 				gamma_is(i, d_prime) = log_gamma_i;
% % 			end
% % 		end
% % 		
% % 		% Summing all probabilities
% % 		for j = 1 : num_contexts
% % 			for d = 1 : D
% % 				log_sum = -Inf;
% % 				for i = 1 : num_contexts
% % 					for d_prime = 1 : D
% % 						if t_beta+d_prime > t_next
% % 							log_beta_tilde = -Inf;
% % 						else
% % 							t_plus_dp = min(t_beta+d_prime, t_next);
% % 							log_beta_tilde = log_beta(t_plus_dp, i, d_prime);
% % 						end
% % 						log_sum = brml.logsumexp([log_sum; log(p_trans(i, j)) + gamma_is(i, d_prime) + gamma_js(j, d) ...
% % 							+ emissions(i, d_prime) + log_beta_tilde]);
% % 					end
% % 				end
% % 				log_beta(t_beta, j, d) = log_sum;
% % 			end
% % 		end		
% % 	end

% end

end

% 		% Time variable
% 		t_p1 = min(t_beta+1, t_next);
% 		for d = 1 : D
% 			% Time variable
% 			t_min_d_p1 = max(1, t_beta-d+1);
% 			for j = 1 : num_contexts
% 				% Sum variable
% 				log_sum = 0;
% 				for d_prime = 1 : D
% 					% Time variables
% 					t_plus_dp = min(t_beta+d_prime, t_next);
% 					time_j = abs(time(t_min_d_p1) - time(t));
% 					time_i = abs(time(t_p1) - time(t_plus_dp));
% 					
% 					if time_j == 0
% 						time_j = 0.5;
% 					end
% 					if time_i == 0
% 						time_i = 0.5;
% 					end
% 					
% 					% Observation probability
% 					log_emission = -emissionMTGP(signal(t_p1 : t_plus_dp, :), time(t_p1 : t_plus_dp), ...
% 						feature_model, gp_model);
% 					
% 					% Gamma J
% 					gamma_j = gamma_model{j};
% 					log_gamma_j = log(gampdf(time_j, gamma_j(1), gamma_j(2)));
% 					
% 					for i = 1 : num_contexts
% 						% Gamma I
% 						gamma_i = gamma_model{i};
% 						log_gamma_i = log(gampdf(time_i, gamma_i(1), gamma_i(2)));
% 						
% 						% Transition probability
% 						trans_ij = p_trans(i, j);
% 						log_trans = log(trans_ij) + log_gamma_i + log_gamma_j;
% 						
% 						if t_plus_dp > t_next
% 							log_beta_tilde = -Inf;
% 						else
% 							log_beta_tilde = log_beta(t_plus_dp, i, d_prime);
% 						end
% 						
% 						log_sum = brml.logsumexp([log_sum; log_beta_tilde + log_trans + log_emission(i)]);
% 						
% 					end
% 				end
% 				log_beta(t_beta, j, d) = log_sum;
% 			end
% 		end
