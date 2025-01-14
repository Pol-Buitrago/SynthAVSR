% Define the filename for 'V' data
file_V = 'wer_results_A.txt';
color_V = '#FFDAB9';  % Color para la modalidad de audio
line_styles_V = {'-', '--'};  % Estilos de línea

% Inicializar la figura con una relación de aspecto más ancha
figure('Position', [100, 100, 1300, 430]);  % Figura ancha
hold on;

% Cargar los datos del archivo 'V'
data_V = readmatrix(file_V);

% Filtrar los datos para Beam Size >= 20 y lenpen = 0.4
filtered_data_V_lenpen_02 = data_V(data_V(:, 2) == 0.4 & data_V(:, 1) >= 1, :);

% Agrupar y calcular el WER promedio para lenpen = 0.4
[unique_beam_size_lenpen_02, ~, idx_lenpen_02] = unique(filtered_data_V_lenpen_02(:, 1));
mean_wer_lenpen_02 = arrayfun(@(x) mean(filtered_data_V_lenpen_02(idx_lenpen_02 == x, 3)), 1:length(unique_beam_size_lenpen_02));

% Graficar los datos para lenpen = 0.4
plot(unique_beam_size_lenpen_02, mean_wer_lenpen_02, 'LineWidth', 3, 'Color', color_V, 'LineStyle', '-'); % lenpen 0.4

% Agregar una línea discontinua en beam_size = 14
xline(14, '--k', 'LineWidth', 2);

% Sombrar el área de beam_size entre 1 y 14 en gris
x_fill = [1, 14, 14, 1];  % Coordenadas x del área sombreada
y_fill = [0, 0, 100, 100];  % Coordenadas y (con rango suficiente para cubrir el WER)
fill(x_fill, y_fill, [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeColor', 'none');  % Área sombreada en gris claro

% Recuadro rojo transparente de Beam Size 30 en adelante
x_fill_red = [30, max(filtered_data_V_lenpen_02(:, 1)), max(filtered_data_V_lenpen_02(:, 1)), 30];
fill(x_fill_red, y_fill, [0.8, 0.9, 1], 'FaceAlpha', 0.2, 'EdgeColor', 'none');  % Área sombreada en rojo

% Agregar líneas alrededor del área roja para resaltarla
line([30, 30], [ylim], 'Color', [0.2, 0.3, 1], 'LineStyle', '--', 'LineWidth', 2); % Línea vertical izquierda

% Encontrar los valores min y max de WER para Beam Size entre 20 y 150
valid_range = filtered_data_V_lenpen_02(filtered_data_V_lenpen_02(:, 1) >= 20 & filtered_data_V_lenpen_02(:, 1) <= 150, :);
min_wer = min(valid_range(:, 3));
max_wer = max(valid_range(:, 3));

% Calcular la diferencia entre los valores máximo y mínimo de WER
wer_difference = max_wer - min_wer;

% Establecer los límites del eje Y para el WER basado en los valores mínimo y máximo
ylim([min_wer - 0.15, max_wer + 0.15]);

% Etiquetas y título en inglés con fuentes más grandes
xlabel('Beam Size', 'FontSize', 20, 'FontWeight', 'bold');
ylabel('WER', 'FontSize', 20, 'FontWeight', 'bold');
title('WER vs Beam Size for Length Penalty = 1 | Audio Modality', 'FontSize', 24, 'FontWeight', 'bold');

% Leyenda con un marcador en naranja oscuro para la configuración elegida
h1 = plot(NaN, NaN, 's', 'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerEdgeColor', 'none', 'MarkerSize', 10); % Cuadrado gris claro
h2 = plot(NaN, NaN, 'LineWidth', 3, 'Color', color_V, 'LineStyle', '-'); % lenpen 0.4
h3 = plot(NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', '#FF8C00'); % Punto naranja oscuro
h4 = plot(NaN, NaN, 's', 'MarkerSize', 10, 'MarkerEdgeColor', [0.2, 0.3, 1], 'MarkerFaceColor', [0.8, 0.9, 1]); % Cuadrado azul claro

% Crear la leyenda
legend([h2, h1, h3, h4], {'Length Penalty = 1', 'Suboptimal Performance', 'Chosen Configuration', 'Computationally Expensive'}, 'Location', 'northwest', 'FontSize', 18);

grid on;

% Personalizar la cuadrícula para una mejor visibilidad
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.5); % Líneas de la cuadrícula discontinuas y menos intensas

% Establecer los límites de los ejes para hacer el gráfico más claro
xlim([min(filtered_data_V_lenpen_02(:, 1)), max(filtered_data_V_lenpen_02(:, 1))]);

% Añadir algo de espacio para evitar que el gráfico quede amontonado
set(gca, 'LooseInset', get(gca, 'TightInset'));

% Aumentar el tamaño de las etiquetas de los ejes
set(gca, 'FontSize', 22);  % Aumentar el tamaño de la fuente para los ticks de los ejes X y Y

% Dibujar líneas discontinuas en el WER mínimo y máximo sin agregarlas a la leyenda
yline(min_wer, '--r', 'LineWidth', 2, 'HandleVisibility', 'off'); % Línea discontinua para el WER mínimo
yline(max_wer, '--r', 'LineWidth', 2, 'HandleVisibility', 'off'); % Línea discontinua para el WER máximo

% Calcular las coordenadas normalizadas para la anotación
y_pos_min = min_wer;
y_pos_max = max_wer;

% Add a point at Beam Size = 17 with dark orange color
beam_size_17_data = filtered_data_V_lenpen_02(filtered_data_V_lenpen_02(:, 1) == 20, :);
plot(beam_size_17_data(:, 1), beam_size_17_data(:, 3), 'o', 'MarkerSize', 8, 'MarkerEdgeColor', '#FF8C00', 'MarkerFaceColor', '#FF8C00', 'HandleVisibility', 'off');

% Añadir el texto de la diferencia cerca de la línea de la flecha
text(42, (max_wer + min_wer) / 2 , sprintf('WER Diff: %.2f', wer_difference), 'FontSize', 18, 'Color', 'k', 'FontWeight', 'bold');

% Guardar la imagen (opcional)
saveas(gcf, 'wer_curve_A_lenpen_04_with_diff.png');
