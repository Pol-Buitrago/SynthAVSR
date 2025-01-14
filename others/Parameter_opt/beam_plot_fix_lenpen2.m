% Define the filename for 'V' data
file_V = 'wer_results_V.txt';
color_V = '#ADD8E6';  % Green color for 'V' data
line_styles_V = {'-', '--'};  % Line styles for positive and negative lenpen

% Initialize the figure with a wider aspect ratio
figure('Position', [100, 100, 1300, 300]);  % Wider figure (apaisada)
hold on;

% Load the data from the 'V' file
data_V = readmatrix(file_V);

% Filter the data for Beam Size >= 20 and lenpen = 0.5
filtered_data_V_lenpen_02 = data_V(data_V(:,2) == 1 & data_V(:,1) >= 1, :);

% Group and calculate the average WER for lenpen = 0.5
[unique_beam_size_lenpen_02, ~, idx_lenpen_02] = unique(filtered_data_V_lenpen_02(:,1));
mean_wer_lenpen_02 = arrayfun(@(x) mean(filtered_data_V_lenpen_02(idx_lenpen_02 == x, 3)), 1:length(unique_beam_size_lenpen_02));

% Plot the data for lenpen = 0.2
plot(unique_beam_size_lenpen_02, mean_wer_lenpen_02, 'LineWidth', 3, 'Color', color_V, 'LineStyle', '-'); % lenpen 0.2

% Add a dashed line at beam_size = 20
xline(20, '--k', 'LineWidth', 2);

% Shade the area from beam_size 1 to 20 in grey
x_fill = [1, 20, 20, 1];  % Define the x coordinates of the shaded area
y_fill = [0, 0, 100, 100];  % Define the y coordinates (set the range high enough to cover the WER)
fill(x_fill, y_fill, [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % Light grey shaded area

% Find the min and max WER values for Beam Size between 20 and 150
valid_range = filtered_data_V_lenpen_02(filtered_data_V_lenpen_02(:,1) >= 20 & filtered_data_V_lenpen_02(:,1) <= 150, :);
min_wer = min(valid_range(:,3));
max_wer = max(valid_range(:,3));

% Calculate the difference between max and min WER
wer_difference = max_wer - min_wer;

% Set the axis limits for WER based on the min and max WER values
ylim([min_wer-1, max_wer+1]);

% Labels and title in English with larger fonts
xlabel('Beam Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('WER', 'FontSize', 20, 'FontWeight', 'bold');
title('WER vs Beam Size for Length Penalty = 1 | Video Modality', 'FontSize', 24, 'FontWeight', 'bold');

% Legend with a gray patch to indicate the shaded area (suboptimal performance)
h1 = plot(NaN, NaN, 's', 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'none', 'MarkerSize', 10); % light gray square patch
h2 = plot(NaN, NaN, 'LineWidth', 3, 'Color', color_V, 'LineStyle', '-'); % lenpen 0.5
legend([h2, h1], {'Length Penalty = 1', 'Suboptimal Performance'}, 'Location', 'northwest', 'FontSize', 18);

grid on;

% Customize grid for better visibility
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5); % Grid lines are dashed and less intense

% Set axes limits to make the plot clearer
xlim([min(filtered_data_V_lenpen_02(:,1)), max(filtered_data_V_lenpen_02(:,1))]);

% Add some padding to make the plot less cluttered
set(gca, 'LooseInset', get(gca, 'TightInset'));

% Increase the size of the tick labels
set(gca, 'FontSize', 22);  % Increase font size for both x and y axis ticks

% Draw dotted lines at the minimum and maximum WER without adding them to the legend
yline(min_wer, '--r', 'LineWidth', 2, 'HandleVisibility', 'off'); % Dotted line for min WER
yline(max_wer, '--r', 'LineWidth', 2, 'HandleVisibility', 'off'); % Dotted line for max WER

% Calculate the normalized coordinates for annotation
y_pos_min = min_wer;
y_pos_max = max_wer;

% Add the difference text next to the arrow line
text(42, (max_wer + min_wer) / 2 - 0.1, sprintf('WER Diff: %.2f', wer_difference), 'FontSize', 18, 'Color', 'k', 'FontWeight', 'bold');

% Save the plot as an image (optional)
saveas(gcf, 'wer_curve_V_lenpen_1_with_diff.png');
