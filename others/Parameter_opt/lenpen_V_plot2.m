% Define the filename for 'V' data
file_V = 'wer_results_V.txt';
color_V = '#2ca02c';  % Green color for 'V' data
line_styles_V = {'-', '--'};  % Line styles for positive and negative lenpen

% Initialize the figure with a wider aspect ratio
figure('Position', [100, 100, 1300, 500]);  % Wider figure (apaisada)
hold on;

% Load the data from the 'V' file
data_V = readmatrix(file_V);

% Filter the data for Beam Size >= 20
filtered_data_V = data_V(data_V(:,1) >= 1, :);

% Split the data into two groups: positive lenpen and negative lenpen
positive_lenpen_data_V = filtered_data_V(filtered_data_V(:,2) > 0, :);
negative_lenpen_data_V = filtered_data_V(filtered_data_V(:,2) < 0, :);

% Group and calculate the average WER for the two lenpen groups
% For positive lenpen
[unique_beam_size_pos_V, ~, idx_pos_V] = unique(positive_lenpen_data_V(:,1));
mean_wer_pos_V = arrayfun(@(x) mean(positive_lenpen_data_V(idx_pos_V == x, 3)), 1:length(unique_beam_size_pos_V));

% For negative lenpen
[unique_beam_size_neg_V, ~, idx_neg_V] = unique(negative_lenpen_data_V(:,1));
mean_wer_neg_V = arrayfun(@(x) mean(negative_lenpen_data_V(idx_neg_V == x, 3)), 1:length(unique_beam_size_neg_V));

% Plot the two lines for 'V' data with thicker lines
plot(unique_beam_size_pos_V, mean_wer_pos_V, 'LineWidth', 3, 'Color', color_V, 'LineStyle', line_styles_V{1}); % positive lenpen
plot(unique_beam_size_neg_V, mean_wer_neg_V, 'LineWidth', 3, 'Color', color_V, 'LineStyle', line_styles_V{2}); % negative lenpen

% Add a dashed line at beam_size = 20
xline(20, '--k', 'LineWidth', 2);

% Shade the area from beam_size 1 to 20 in grey
x_fill = [1, 20, 20, 1];  % Define the x coordinates of the shaded area
y_fill = [0, 0, 100, 100];  % Define the y coordinates (set the range high enough to cover the WER)
fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Shaded area

% Labels and title in English with larger fonts
xlabel('Beam Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('WER', 'FontSize', 20, 'FontWeight', 'bold');
title('WER vs Beam Size for Positive and Negative Length Penalty (V)', 'FontSize', 24, 'FontWeight', 'bold');

% Legend with a gray patch to indicate the shaded area (suboptimal performance)
h1 = plot(NaN, NaN, 's', 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'none', 'MarkerSize', 25); % gray square patch
h2 = plot(NaN, NaN, 'LineWidth', 3, 'Color', color_V, 'LineStyle', line_styles_V{1}); % positive lenpen
h3 = plot(NaN, NaN, 'LineWidth', 3, 'Color', color_V, 'LineStyle', line_styles_V{2}); % negative lenpen
legend([h2, h3, h1], {'Positive Length Penalty V', 'Negative Length Penalty V', 'Suboptimal Performance'}, 'Location', 'northwest', 'FontSize', 18);

grid on;

% Customize grid for better visibility
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5); % Grid lines are dashed and less intense

% Set axes limits to make the plot clearer
xlim([min(data_V(:,1)), max(data_V(:,1))]);
ylim([80, 90]);  % Set WER axis limit to 90

% Add some padding to make the plot less cluttered
set(gca, 'LooseInset', get(gca, 'TightInset'));

% Increase the size of the tick labels
set(gca, 'FontSize', 22);  % Increase font size for both x and y axis ticks

% Save the plot as an image (optional)
saveas(gcf, 'wer_curve_V_beam_lenpen.png');
