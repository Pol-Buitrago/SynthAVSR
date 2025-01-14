% Datos de WER para las modalidades Audio (A) y Audiovisual (AV)
hours_4 = [83, 82];  % WER con ~4 horas de entrenamiento
hours_9 = [30, 29.6];  % WER con ~9 horas de entrenamiento
hours_19 = [28.1, 27.7];  % WER con ~19 horas de entrenamiento
hours_196 = [13.64, 12.8];  % WER con ~196 horas de entrenamiento

% Crear la figura y ajustarla a un formato apaisado
figure;
set(gcf, 'Position', [100, 100, 1300, 400]);  % Tamaño apaisado (ancho, alto)
hold on;

% Graficar las líneas para cada modalidad con colores morado y marrón
h1 = plot([4, 9, 19, 196], [hours_4(1), hours_9(1), hours_19(1), hours_196(1)], '-o', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [0.6, 0.2, 0.8], 'DisplayName', 'Audio (A)', 'Color', [0.6, 0.2, 0.8]);  % Morado
h2 = plot([4, 9, 19, 196], [hours_4(2), hours_9(2), hours_19(2), hours_196(2)], '-o', 'LineWidth', 3, 'MarkerSize', 10, 'MarkerFaceColor', [1, 0.6, 0], 'DisplayName', 'Audiovisual (AV)', 'Color', [1, 0.6, 0]);  % Marrón

% Sombrar el área entre las dos curvas
fill([4, 9, 19, 196, 196, 19, 9, 4], ...
     [hours_4(1), hours_9(1), hours_19(1), hours_196(1), hours_196(2), hours_19(2), hours_9(2), hours_4(2)], ...
     [0.8, 0.8, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.6);  % Color suave para el área sombreada

% Configurar detalles de la gráfica
xlabel('Training Hours', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
ylabel('WER (%)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
title('WER Comparison for Audio (A) and Audiovisual (AV) Modalities as Data Increases', 'FontSize', 18, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
lgd = legend([h1, h2], {'Audio (A)', 'Audiovisual (AV)'}, 'Location', 'northeast', 'FontSize', 14, 'FontWeight', 'bold', 'TextColor', [0.2, 0.2, 0.2]);
lgd.Position = [0.725, 0.76, 0.14, 0.1];  % Ajusta la posición de la leyenda

% Mejorar la presentación
set(gca, 'FontSize', 14, 'FontName', 'Arial', 'GridColor', [0.8, 0.8, 0.8], 'GridAlpha', 0.5);
set(gca, 'Box', 'on', 'LineWidth', 1.5);
grid on;
axis tight;
hold off;

% Guardar la gráfica como un archivo PNG
saveas(gcf, 'WER_hours.png');
