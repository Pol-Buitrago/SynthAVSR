% Define the filename for the Audio modality (A) data
file_A = 'wer_results_A.txt';

% Load the data from the file
data_A = readmatrix(file_A);

% Filter the data where Beam Size is less than 30
filtered_data_A = data_A(data_A(:, 1) < 30, :);

% Extract the WER column (assumed to be the third column) from the filtered data
WER_values = filtered_data_A(:, 3);

% Find the indices of the 50 smallest WER values
[sorted_WER, sorted_indices] = sort(WER_values); % Sort WER values in ascending order
top_50_indices = sorted_indices(1:50);          % Get the indices of the 50 smallest values

% Extract the corresponding rows for the top 50 WER values
top_50_rows = filtered_data_A(top_50_indices, :);

% Display the results
disp('Top 50 smallest WER values with Beam Size < 30 and their corresponding rows:');
disp(array2table(top_50_rows, 'VariableNames', {'Beam_Size', 'Length_Penalty', 'WER'}));

% Save the top 50 rows to a new file (optional)
writematrix(top_50_rows, 'top_50_smallest_WER_A_filtered.txt', 'Delimiter', 'tab');
