function [context_pred, log_prob] = predictContext2(signal, time, model, pred_lag, max_duration)
%PREDICTCONTEXT Summary of this function goes here
%   Detailed explanation goes here

import brml.*

% Feature model
feature_model = model.feature_model;

% HMM Model
p_trans = model.hmm_model;

% Gamma Process Model
gamma_model = model.gamma_model;

% GP Model
baseline_model = model.baseline_model;
context_model = model.context_model;

% Constants
num_contexts = length(gamma_model);
time_length = length(time);
D = max_duration;

% Forward-Backward Prediction
log_alpha = zeros(time_length, num_contexts, max_duration);
log_beta = zeros(pred_lag, num_contexts, max_duration);

context_pred = zeros(time_length, 1);

log_prob = 0;

% Initial alpha and beta
t = 1;
d = 1;
log_alpha(t, :, d) = -emissionMTGP(signal(t, :), time(t), feature_model, gp_model);

[~, context_pred(t)] = max(log_alpha(t, :, d));
disp(context_pred(t));

% for t = 2 : time_length
% 	[~, context_pred(t)] = min(emissionMTGP(signal(t, :), time(t), feature_model, gp_model));
% end

t_next = min(t + pred_lag, length(time));

% Recurse
for t = 2 : time_length
	
	
	for d = 1 : D
		t_prev = max(1, t-d+1);
		t_p_prev = max(1, t-d);
		
		log_emission = -emissionMTGP(signal(t_prev : t, :), time(t_prev : t), ...
			feature_model, gp_model);
		
		time_j = abs(time(t) - time(t_prev));
		if time_j == 0
			time_j = 0.5;
		end
		
		for j = 1 : num_contexts
			log_sum = 0;
			
			gamma_j = gamma_model{j};
			log_gamma_j = log(gampdf(time_j, gamma_j(1), gamma_j(2)));
			
% 			if log_gamma_j == -inf
% 				log_gamma_j = 0;
% 			end

			next_state = 1 : num_contexts;
			next_state(j) = [];

			for i = next_state
				gamma_i = gamma_model{i};
				trans_ji = p_trans(j, i);
				
				for d_prime = 1 : D
					t_pTilde = max(1, t - d - d_prime + 1);
					
					time_i = abs(time(t_pTilde) - time(t_p_prev));
					if time_i == 0
						time_i = 0.5;
					end
					
					log_gamma_i = log(gampdf(time_i, gamma_i(1), gamma_i(2)));
					
% 					if log_gamma_i == -inf
% 						log_gamma_i = 0;
% 					end
					
					log_trans = log(trans_ji) + log_gamma_i + log_gamma_j;
					log_sum = brml.logsumexp([log_sum; log_alpha(t_p_prev, i, d_prime) + log_trans]);
				end
			end
			log_alpha(t, j, d) = log_emission(j) + log_sum;
		end
	end
	[~, context_pred(t)] = max(sum(log_alpha(t, :, :), 3));
	disp(context_pred(t));
end

end

