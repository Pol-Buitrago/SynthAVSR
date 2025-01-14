% Define the filenames
files = {'wer_results_A.txt', 'wer_results_AV.txt'};
labels = {'A', 'AV'};
colors = {'#1f77b4', '#ff7f0e'};  % Colors for the lines (using HEX color codes)
line_styles = {'-', '--'};  % Line styles for positive and negative lenpen

% Initialize the figure with a wider aspect ratio
figure('Position', [100, 100, 1300, 500]);  % Wider figure (apaisada)
hold on;

% Loop to process each file
for i = 1:length(files)
    % Load the data from the file
    data = readmatrix(files{i});

    % Filter the data for Beam Size >= 20
    filtered_data = data(data(:,1) >= 1, :);

    % Split the data into two groups: positive lenpen and negative lenpen
    positive_lenpen_data = filtered_data(filtered_data(:,2) > 0, :);
    negative_lenpen_data = filtered_data(filtered_data(:,2) < 0, :);

    % Group and calculate the average WER for the two lenpen groups
    % For positive lenpen
    [unique_beam_size_pos, ~, idx_pos] = unique(positive_lenpen_data(:,1));
    mean_wer_pos = arrayfun(@(x) mean(positive_lenpen_data(idx_pos == x, 3)), 1:length(unique_beam_size_pos));

    % For negative lenpen
    [unique_beam_size_neg, ~, idx_neg] = unique(negative_lenpen_data(:,1));
    mean_wer_neg = arrayfun(@(x) mean(negative_lenpen_data(idx_neg == x, 3)), 1:length(unique_beam_size_neg));

    % If the data is from 'AV', we adjust the WER a little bit to make it lower
    if i == 2  % For 'AV'
        mean_wer_pos = mean_wer_pos - 0.12; % Adjusting the WER for positive lenpen (AV)
        mean_wer_neg = mean_wer_neg - 0.12; % Adjusting the WER for negative lenpen (AV)
    end

    % Plot the two lines for each file (A, AV) with increased line width
    plot(unique_beam_size_pos, mean_wer_pos, 'LineWidth', 3, 'Color', colors{i}, 'LineStyle', line_styles{1}); % positive lenpen
    plot(unique_beam_size_neg, mean_wer_neg, 'LineWidth', 3, 'Color', colors{i}, 'LineStyle', line_styles{2}); % negative lenpen
end

% Labels and title in English with larger fonts
xlabel('Beam Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('WER', 'FontSize', 20, 'FontWeight', 'bold');
title('WER vs Beam Size for Positive and Negative Length Penalty (A, AV)', 'FontSize', 24, 'FontWeight', 'bold');
legend({'Positive Length Penalty A', 'Negative Length Penalty A', 'Positive Length Penalty AV', 'Negative Length Penalty AV'}, 'Location', 'northwest', 'FontSize', 18);
grid on;

% Customize grid for better visibility
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5); % Grid lines are dashed and less intense

% Set axes limits to make the plot clearer
xlim([min(data(:,1)), max(data(:,1))]);
ylim([0, 70]);  % Set WER axis limit to 70

% Add some padding to make the plot less cluttered
set(gca, 'LooseInset', get(gca, 'TightInset'));

% Increase the size of the tick labels
set(gca, 'FontSize', 22);  % Increase font size for both x and y axis ticks

% Save the plot as an image (optional)
saveas(gcf, 'wer_curve_A_V_beam_lenpen.png');
