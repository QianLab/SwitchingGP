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

feature_model = importdata('../Results/MTGP/feature_model.mat');

hmm_model = importdata('../Results/MTGP/hmm_model.mat');
hmm_model = hmm_model + 1;
hmm_model = hmm_model ./ sum(hmm_model);

gamma_model = importdata('../Results/MTGP/gamma_model.mat');
baseline_model = importdata('../Results/MTGP_yeah/baseline_model.mat');
context_model = importdata('../Results/MTGP_yeah/context_model.mat');

model = struct;
model.feature_model = feature_model;
model.hmm_model = hmm_model;
model.gamma_model = gamma_model;
model.baseline_model = baseline_model;
model.context_model = context_model;

max_duration = 50;

for i = 1 : length(signals_train)
	signal = signals_train{i};
	time = 1 : size(signal, 1);
	true_context = context_train{i};
	
	[predicted_context, ~] = predictContext(signal, time, model, pred_lag, max_duration);
	disp(sum(predicted_context == true_context)/length(true_context));
	% Remark: 66% accuracy just single point likelihood (105/302 wrong)
end

% for i 
% 
% [Xtildes, UiTildes, A, Bjs, C, sigma2, sTrans, pStGivenX1Ts, logLikelihoods, initD, initPStGivenX1Ts] = inputGroupSARLearn(Xis, Uis, Lx, Lu, S, Tskip, maxIterations, imputeParams, PStGivenX1TInit, Dinit)