function [log_alpha_next, log_trans_next]  = MTGPAlpha(log_alpha, t, signal, time, time_indices, feat_indices, max_duration, ...
    feature_model, gp_model, gamma_model, hmm_model, sub_num, feature_lengths, y_sample)

D = max_duration;
num_contexts = length(gamma_model);
log_alpha_next = zeros(num_contexts, D);
log_trans_next = zeros(num_contexts, D);

p_trans = hmm_model;

% Forward step
for d = 1 : min(D, t-1)
  % Time Variables
  t_min_d_p1 = max(1, t-d+1);
  t_min_d = max(1, t-d);
  time_j = abs(time(t) - time(t_min_d_p1));

  % Emission
  ind = sum(feature_lengths(end-d+1 : end));
  
  time_indices_tmp = time_indices(end - ind + 1 : end);
  time_indices_tmp = time_indices_tmp - min(time_indices_tmp) + 1;
  
  if isempty(y_sample)
    log_emission = -emissionMTGPSelect(signal(t_min_d_p1 : t, :), time(t_min_d_p1 : t), time_indices_tmp, ...
        feat_indices(end - ind + 1 : end), feature_model, gp_model, sub_num, y_sample);
  else
    log_emission = -emissionMTGPSelect(signal(t_min_d_p1 : t-1, :), time(t_min_d_p1 : t), time_indices_tmp, ...
        feat_indices(end - ind + 1 : end), feature_model, gp_model, sub_num, y_sample);
  end
  for j = 1 : num_contexts
    % Sum variable
    log_sum = 0;
    
    % Gamma J
    gamma_j = gamma_model{j};
    log_gamma_j = log(gampdf(time_j, gamma_j(1), gamma_j(2)));

    next_state = 1 : num_contexts;
    next_state(j) = [];
    for i = next_state
      % Gamma I
      gamma_i = gamma_model{i};
      
      for d_prime = 1 : D
        % Time variables
        t_min_d_dp_p1 = max(1, t - d - d_prime + 1);
        time_i = abs(time(t_min_d_dp_p1) - time(t_min_d));
        
        % Gamma I
        log_gamma_i = log(gampdf(time_i, gamma_i(1), gamma_i(2)));

        % Transition probability
        trans_ji = p_trans(j, i);
        log_trans = log(trans_ji) + log_gamma_i + log_gamma_j;
        
        if (t-d < 0)
          log_alpha_tilde = -Inf;
        else
          log_alpha_tilde = log_alpha(t_min_d, i, d_prime);
        end
          
        log_sum = brml.logsumexp([log_sum; log_alpha_tilde + log_trans]);
      end
    end
    log_alpha_next(j, d) = log_emission(j) + log_sum;
    log_trans_next(j, d) = log_trans;
  end
end

end
