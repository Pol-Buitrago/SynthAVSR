% Datos de WER para la modalidad visual (V)
datasets = {'LIP-RTVE', 'CMU-MOSEAS_{es}', 'MuAViC_{es}', 'Full Corpus'};  % Nombres de los datasets
real_data = [90.4, 90.76, 102.02, 94.92];  % WER con datos reales (V)
synth_data = [97.46, 97.02, 101.96, 99.16];  % WER con datos sintéticos (V)
synth_real_data = [68.09, 67.14, 98.06, 79.5];  % WER con datos sintéticos + reales (V)

% Crear la figura con un tamaño más apaisado
figure;
set(gcf, 'Position', [100, 100, 1200, 400]);  % Tamaño apaisado (ancho, alto)

% Crear las posiciones para las barras
barWidth = 0.2;  % Ancho de las barras
r1 = 1:length(datasets);  % Posiciones de las barras para datos reales
r2 = r1 + barWidth;  % Posiciones de las barras para datos sintéticos
r3 = r2 + barWidth;  % Posiciones de las barras para datos sintéticos + reales

% Dibujar las barras con colores más suaves y bordes más definidos
b1 = bar(r1, real_data, barWidth, 'FaceColor', [0.2, 0.6, 1], 'EdgeColor', 'k', 'LineWidth', 1.5);  % Barras para Real
hold on;
b2 = bar(r2, synth_data, barWidth, 'FaceColor', [0.4, 0.8, 0.4], 'EdgeColor', 'k', 'LineWidth', 1.5);  % Barras para Synth
b3 = bar(r3, synth_real_data, barWidth, 'FaceColor', [1, 0.4, 0.4], 'EdgeColor', 'k', 'LineWidth', 1.5);  % Barras para Synth+Real

% Configurar detalles de la gráfica
set(gca, 'XTick', r1 + barWidth, 'XTickLabel', datasets, 'FontSize', 14, 'FontName', 'Arial');
xlabel('Dataset', 'FontWeight', 'bold', 'FontSize', 16);
ylabel('WER (%)', 'FontWeight', 'bold', 'FontSize', 16);
title('WER Comparison: Training Lipreading with Real, Synthetic, and Combined (Synth + Real) Data', 'FontSize', 18, 'FontWeight', 'bold');

% Mejorar el fondo y los ejes
set(gca, 'Color', [1, 1, 1], 'Box', 'on', 'GridLineStyle', '--', 'GridColor', [0.5, 0.5, 0.5], 'LineWidth', 1.2);
grid on;
axis tight;

% Ajuste de límites del eje Y
ylim([0, max([real_data, synth_data, synth_real_data]) * 1.1]);  % Aumentar el límite Y en un 10%

% Ajuste de límites del eje X para dejar margen en los lados
xlim([0.5, length(datasets) + 4]);  % Añadir margen en los lados de la gráfica

% Mejorar los márgenes y las etiquetas
set(gca, 'TickLength', [0.02, 0.025]);
ax = gca;
ax.XColor = 'k';
ax.YColor = 'k';

% Configurar la leyenda y moverla ligeramente
lgd = legend([b1, b2, b3], {'Real Data (V)', 'Synthetic Data (V)', 'Synth + Real Data (V)'}, 'Location', 'northeast', 'FontSize', 14, 'FontWeight', 'bold');
lgd.Position = [0.715, 0.75, 0.15, 0.12];  % Ajusta la posición de la leyenda

% Calcular la diferencia entre 'Real' y 'Synth + Real'
differences = real_data - synth_real_data;

% Imprimir las diferencias debajo de la leyenda, en una posición fija (0.6, 0.6)
y_position = 71;  % Posición Y inicial
for i = 1:length(differences)
    % Texto con el símbolo Δ para la diferencia
    text(6.2, y_position, sprintf('\\DeltaWER (Real vs Synth+Real) for %s: %.2f (↓)', datasets{i}, differences(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 14, 'FontWeight', 'bold');
    y_position = y_position - 20;  % Mover la posición Y para la siguiente diferencia
end

% Guardar la figura en un archivo PNG
saveas(gcf, 'WER_comparison_Visual_Modality.png');

