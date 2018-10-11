function [signals_train, signals_test, mean_train, variance_train, freq] = importUCIHAR()

%% Import Data
disp('Importing Data');

freq = 1000/50; % ms

signals_train = importdata('../Data/UCI HAR Dataset/train/X_train.txt');
signals_test = importdata('../Data/UCI HAR Dataset/test/X_test.txt');

subject_train = importdata('../Data/UCI HAR Dataset/train/subject_train.txt');
subject_test = importdata('../Data/UCI HAR Dataset/test/subject_test.txt');

context_train_raw = importdata('../Data/UCI HAR Dataset/train/y_train.txt');
context_test_raw = importdata('../Data/UCI HAR Dataset/test/y_test.txt');

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

mean_train = mean(signals_train);
variance_train = var(signals_train);

signals_train = (signals_train - mean_train) / variance_train;

%% Divide Data Based on Different Subjects
disp('Formatting Data by Subjects');

num_subjects = max(subject_train);
num_contexts = max(context_train_raw);
train_subjects = 1 : num_subjects;

s_train = cell(num_subjects, 1);
context_train = cell(num_subjects, 1);
for i = 1 : length(s_train)
	s_train{i} = signals_train(subject_train == i, :);
	context_train{i} = context_train_raw(subject_train == i);
end

empty_cells = cellfun(@isempty, s_train);
s_train(empty_cells) = [];
context_train(empty_cells) = [];
train_subjects(empty_cells) = [];

test_subjects = 1 : num_subjects;
s_test = cell(num_subjects, 1);
context_test = cell(num_subjects, 1);
for i = 1 : length(s_test)
	s_test{i} = signals_test(subject_test == i, :);
	context_test{i} = context_test_raw(subject_test == i);
end

empty_cells = cellfun(@isempty, s_test);
s_test(empty_cells) = [];
context_test(empty_cells) = [];
test_subjects(empty_cells) = [];

signals_train = s_train;
signals_test = s_test;

%% Divide by contexts
disp('Dividing by Contexts');

s_train = cell(num_contexts, length(signals_train));
s_test = cell(num_contexts, length(signals_test));

for j = 1 : num_contexts	
	for i = 1 : length(signals_train)
		current_context = context_train{i} == j;
		disjoints = (current_context ~= 0)';
		
		dis1 = strfind([0 disjoints], [0 1]);
		dis2 = strfind([disjoints 0], [1 0]);
		context_signals = arrayfun(@(x,y) signals_train{i}(x:y, :), dis1, dis2, 'un', 0);
		
		s_train{j, i} = context_signals';
	end
	
	for i = 1 : length(signals_test)
		current_context = context_test{i} == j;
		disjoints = (current_context ~= 0)';

		dis1 = strfind([0 disjoints], [0 1]);
		dis2 = strfind([disjoints 0], [1 0]);
		context_signals = arrayfun(@(x,y) signals_test{i}(x:y, :), dis1, dis2, 'un', 0);
		
		s_test{j, i} = context_signals';
	end
end

signals_train = s_train;
signals_test = s_test;

clear i j s_train s_test dis1 dis2 disjoints context_signals current_context;

end
