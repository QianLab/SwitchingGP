tic;
clear; close all; clc;

% Remark: For state prediction, need to determine a lag term. 0 lag =
% filtering, d lag = smoothing. Maybe around d=5?

addpath(genpath('../'))

%% Import Data
disp('Importing Data');

freq = 1000/50; % ms

signals_train = importdata('../Data/UCI HAR Dataset/train/X_train.txt');
signals_test = importdata('../Data/UCI HAR Dataset/test/X_test.txt');

subject_train = importdata('../Data/UCI HAR Dataset/train/subject_train.txt');
subject_test = importdata('../Data/UCI HAR Dataset/test/subject_test.txt');

raw_context_train = importdata('../Data/UCI HAR Dataset/train/y_train.txt');
raw_context_test = importdata('../Data/UCI HAR Dataset/test/y_test.txt');

raw_train_directory = dir('../Data/UCI HAR Dataset/train/Inertial Signals/*.txt');
raw_test_directory = dir('../Data/UCI HAR Dataset/test/Inertial Signals/*.txt');

signals_train_raw = [];
for i = 1 : length(raw_train_directory)
	curr_raw = importdata(['../Data/UCI HAR Dataset/train/Inertial Signals/', raw_train_directory(1).name]);
	signals_train_raw = cat(3, signals_train_raw, curr_raw);
end
	
signals_test_raw = [];
for i = 1 : length(raw_test_directory)
	curr_raw = importdata(['../Data/UCI HAR Dataset/test/Inertial Signals/', raw_test_directory(1).name]);
	signals_test_raw = cat(3, signals_test_raw, curr_raw);
end

clear curr_raw i;

%% Divide Data Based on Different Subjects
disp('Formatting Data');

num_subjects = max(subject_train);
num_contexts = max(raw_context_train);

train_subjects = 1 : num_subjects;
signals_training = cell(num_subjects, 1);
context_train = cell(num_subjects, 1);
for i = 1 : length(signals_training)
	signals_training{i} = signals_train(subject_train == i, :);
	context_train{i} = raw_context_train(subject_train == i);
end

empty_cells = cellfun(@isempty, signals_training);
signals_training(empty_cells) = [];
context_train(empty_cells) = [];
train_subjects(empty_cells) = [];

test_subjects = 1 : num_subjects;
signals_testing = cell(num_subjects, 1);
context_test = cell(num_subjects, 1);
for i = 1 : length(signals_testing)
	signals_testing{i} = signals_test(subject_test == i, :);
	context_test{i} = raw_context_test(subject_test == i);
end

empty_cells = cellfun(@isempty, signals_testing);
signals_testing(empty_cells) = [];
context_test(empty_cells) = [];
test_subjects(empty_cells) = [];

signals_train = signals_training;
signals_test = signals_testing;
clear empty_cells signals_training signals_testing raw_context_train raw_context_test raw_train_directory raw_test_directory i subject_train subject_test;

%% Predict context using learned models
disp('Predicting Contexts');

pred_lag = 5;
threshold = 10;

feature_model = importdata('../Results/MTGP/feature_model.mat');
feature_model.means(end) = [];
feature_model.pca_coeffs(end) = [];

hmm_model = importdata('../Results/MTGP/hmm_model.mat');
hmm_model = hmm_model + 1;
hmm_model(:, end) = [];
hmm_model(end, :) = [];
hmm_model = hmm_model ./ sum(hmm_model);

gamma_model = importdata('../Results/MTGP_mattern5/gamma_model.mat');
gamma_model(end) = [];

gp_model = importdata('../Results/MTGP_mattern5/gp_model.mat');
gp_model.coeffs(end) = [];

model = struct;
model.feature_model = feature_model;
model.hmm_model = hmm_model;
model.gamma_model = gamma_model;
model.gp_model = gp_model;
max_duration = 1;

all_predictions = cell(length(signals_train), 1);
all_alpha_prob = cell(length(signals_train), 1);
all_energy_costs= cell(length(signals_train), 1);
all_entropies = cell(length(signals_train), 1);
all_feature_sets = cell(length(signals_train), 1);
errors = cell(length(signals_train), 1);

averaged_error = 0;
for n = 1 : 100
	outlier_errors = cell(length(signals_train), 1);
	for i = 1 : length(signals_train)
		true_outliers = context_train{i};
		outlier_pred = randi([1, 6], size(true_outliers));

		outlier_pred = (outlier_pred == 6);
		true_outliers = (true_outliers == 6);

		outlier_errors{i} = (outlier_pred ~= true_outliers);
	end
	all_err = vertcat(outlier_errors{:});
	total_error = sum(all_err)/length(all_err);
	averaged_error = averaged_error + total_error;
end

averaged_error = averaged_error/100;
	
for i = 1 : length(signals_train)
	signal = signals_train{i};
	time = 1 : size(signal, 1);
	true_context = context_train{i};
	
	[context_pred, alpha_prob, entropies] = unknown_context(signal, time, model, pred_lag, max_duration, i, threshold);
    
	errors{i} = (context_pred ~= true_context);
	all_predictions{i} = context_pred;
	all_alpha_prob{i} = alpha_prob;    
	all_entropies{i} = entropies;
	
end

for i = 1 : length(signals_train)
	outlier_pred = all_predictions{i};
	true_outliers = context_train{i};
	
	outlier_pred = (outlier_pred == 6);
	true_outliers = (true_outliers == 6);
	
	outlier_errors{i} = (outlier_pred ~= true_outliers);
end

save('../Results/MTGP_mattern5/context_predictions.mat', 'all_predictions');
save('../Results/MTGP_mattern5/context_entropies.mat', 'all_entropies');
save('../Results/MTGP_mattern5/context_probabilities.mat', 'all_alpha_prob');
save('../Results/MTGP_mattern5/outlier_errors.mat', 'outlier_errors');

data_ratio = 4;
num_features = 10;
rank_approx = num_features;
num_obs = ones(num_features, 1);

time = 1 : size(signals_train{1}, 1);

time_obs = downsample(time', data_ratio);
signal_obs = signals_train{1}(time_obs, :);

task_index_obs = kron(1 : size(signal_obs, 2), ones(1, size(signal_obs, 1)))';
time_index_obs = repmat(1 : size(signal_obs, 1), 1, size(signal_obs, 2))';

signal_obs_in = signal_obs(:);
time_pred = time';
signal_true = signals_train{1}(time_pred, :);

data_obs = {cov_contexts, time_obs, signal_obs_in, num_features, rank_approx, num_obs, task_index_obs, time_index_obs};

log_thetas = gp_model.coeffs;

signal_pred = cell(num_contexts, 1);
Vpred = cell(num_contexts, 1);
for j = 1 : num_contexts
	[signal_pred{j}, Vpred{j}] = predict_popmtgp_all_tasks(log_thetas{j}, data_obs, time_pred);
end

% Average
state_prob = brml.condexp(alpha_prob')';

signal_pred_avg = zeros(length(time), num_features);
Vpred_avg = zeros(length(time), num_features);
for j = 1 : num_contexts
	signal_pred_avg = signal_pred_avg + signal_pred{j} .* state_prob(:, j);
	Vpred_avg = Vpred_avg + Vpred{j} .* state_prob(:, j).^2;
end

Ypred_up = signal_pred_avg + 1.96 * sqrt(Vpred_avg);
Ypred_down = signal_pred_avg - 1.96 * sqrt(Vpred_avg);

% Plot
% figure('units', 'normalized', 'outerposition', [0 0 1 1], 'visible', 'off');
figure('units', 'normalized', 'outerposition', [0 0 1 1]);
suptitle(['Predictions Context ' num2str(j), ' Trial ' num2str(i)]);
for p = 1 : num_features
	subplot(5, 2, p);
	hold on; box on;

	% Prediction and confidence intervals
	conf_color = [0.95 0.95 0.95];
	ciplot(Ypred_up(:, p), Ypred_down(:, p), time_pred * freq, conf_color);
	plot(time_pred * freq, signal_pred_avg(:, p));
	
	% Data points
	scatter(time_obs * freq, signal_obs(:, p), 60, 'd', 'red', 'filled');
	scatter(time_pred * freq, signal_true(:, p), 'o', 'blue');
	
	xlabel('Time (ms)'), ylabel(['Feature ' num2str(p)]);
end
saveFigure('../Results/MTGP/', ['Context ' num2str(j), ' Trial ' num2str(i)]);

% for i 
% 
% [Xtildes, UiTildes, A, Bjs, C, sigma2, sTrans, pStGivenX1Ts, logLikelihoods, initD, initPStGivenX1Ts] = inputGroupSARLearn(Xis, Uis, Lx, Lu, S, Tskip, maxIterations, imputeParams, PStGivenX1TInit, Dinit)
