tic;
clear; close all; clc;

addpath(genpath('../'));

%% Info
% 1 WALKING
% 2 WALKING_UPSTAIRS
% 3 WALKING_DOWNSTAIRS
% 4 SITTING
% 5 STANDING
% 6 LAYING

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
disp('Formatting Data by Subjects');

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

%% Divide by contexts
disp('Dividing by Contexts');

signals_training = cell(num_contexts, 1);
signals_testing = cell(num_contexts, 1);

for j = 1 : num_contexts
	signals_training{j} = {};
	signals_testing{j} = {};
	
	for i = 1 : length(signals_train)
		current_context = context_train{i} == j;
		disjoints = (current_context ~= 0)';
		
		dis1 = strfind([0 disjoints], [0 1]);
		dis2 = strfind([disjoints 0], [1 0]);
		context_signals = arrayfun(@(x,y) signals_train{i}(x:y, :), dis1, dis2, 'un', 0);
		
		signals_training{j} = [signals_training{j} context_signals];
	end
	
	for i = 1 : length(signals_test)
		current_context = context_test{i} == j;
		disjoints = (current_context ~= 0)';

		dis1 = strfind([0 disjoints], [0 1]);
		dis2 = strfind([disjoints 0], [1 0]);
		context_signals = arrayfun(@(x,y) signals_test{i}(x:y, :), dis1, dis2, 'un', 0);

		signals_testing{j} = [signals_testing{j} context_signals];
	end
end

signals_train = signals_training;
signals_test = signals_testing;

clear i j signals_training signals_testing dis1 dis2 disjoints;

%% Apply PACE
disp('Applying PACE');

paceOptions = setOptions('selection_k', 10, 'maxk', 10, 'ngrid', 75, 'ngrid1', ...
	55, 'numBins', 0, 'ntest1', 55);

fpca_coeffs = cell(num_contexts, 1);

for j = 1 : num_contexts
	signals = signals_train{j};
	signals = cellfun(@transpose, signals, 'UniformOutput', false);
	
	time = cell(1, length(signals));
	
	for i = 1 : length(signals)
		time{i} = ones(size(signals{i}, 1), 1) .* (1 : size(signals{i}, 2));
	end
	
	fpcaBasis = FPCA(signals, time, paceOptions);
end