function [total_nl, total_gradnl] = nmargl_mtgpnew(logtheta, logtheta_all, cov_func, times, signals, ...
				      num_features, rank_approx, task_index, time_index, deriv_range)
%NMARGL_MTGPNEW Summary of this function goes here
%   Detailed explanation goes here

total_nl = 0;
total_gradnl = 0;

signal_indices = randperm(length(signals));
% signal_indices = 1 : length(signals);

batch_size = 21;
count = 1;
for i = signal_indices
	signal = signals{i};
	time = times{i};
	num_obs = ones(length(signal), 1);
	
	if count <= batch_size
		[nl, gradnl] = nmargl_mtgp(logtheta, logtheta_all, cov_func, time, signal, num_features, rank_approx, num_obs, task_index{i}, time_index{i}, deriv_range);
		total_nl = total_nl + nl;
		total_gradnl = total_gradnl + gradnl;
	else
		nl = nmargl_mtgp(logtheta, logtheta_all, cov_func, time, signal, num_features, rank_approx, num_obs, task_index{i}, time_index{i}, deriv_range);
		total_nl = total_nl + nl;
	end
	count = count + 1;
end

disp('Computed Likelihood');

end

