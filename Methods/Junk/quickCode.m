test1 = zeros(1, max(subject_train));

for i = 1 : max(subject_train)
	what = subject_train == i;
	what1 = [what; 0];
	what2 = [0; what];
	
	test1(i) = sum(abs(what1 - what2));
	
end

test2 = zeros(1, max(subject_test));

for i = 1 : max(subject_test)
	what = subject_test == i;
	what1 = [what; 0];
	what2 = [0; what];
	
	test2(i) = sum(abs(what1 - what2));
	
end