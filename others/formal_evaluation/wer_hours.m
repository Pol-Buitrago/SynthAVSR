% Datos de WER para las modalidades Audio (A) y Audiovisual (AV)
hours_4 = [82.93, 82.28];      % WER con ~4 horas de entrenamiento
hours_9 = [30, 29.66];    % WER con ~9 horas de entrenamiento
hours_19 = [28.19, 27.69]; % WER con ~19 horas de entrenamiento
hours_196 = [13.64, 12.8]; % WER con ~196 horas de entrenamiento

% Calcular la mejora porcentual de AV respecto a A
improvement_4 = ((hours_4(1) - hours_4(2)) / hours_4(1)) * 100;
improvement_9 = ((hours_9(1) - hours_9(2)) / hours_9(1)) * 100;
improvement_19 = ((hours_19(1) - hours_19(2)) / hours_19(1)) * 100;
improvement_196 = ((hours_196(1) - hours_196(2)) / hours_196(1)) * 100;

% Vector de horas y mejoras porcentuales
training_hours = [4, 9, 19, 196];
improvements = [improvement_4, improvement_9, improvement_19, improvement_196];

% Crear la figura y ajustarla a un formato apaisado
figure;
set(gcf, 'Position', [100, 100, 1300, 400]);  % Tamaño apaisado (ancho, alto)
hold on;

% Graficar la mejora porcentual con una línea y marcadores
plot(training_hours, improvements, '-o', 'LineWidth', 3, 'MarkerSize', 10, ...
    'MarkerFaceColor', [0.2, 0.6, 0.8], 'Color', [0.2, 0.6, 0.8]);

% Configurar detalles de la gráfica
xlabel('Training Hours', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
ylabel('Improvement (%)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
title('Percentage Improvement of Audiovisual (AV) Compared to Audio (A)', 'FontSize', 18, ...
    'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
grid on;

% Ajustar el eje y añadir leyenda
set(gca, 'FontSize', 14, 'FontName', 'Arial', 'GridColor', [0.8, 0.8, 0.8], 'GridAlpha', 0.5);
set(gca, 'Box', 'on', 'LineWidth', 1.5);
axis tight;
hold off;

% Guardar la gráfica como un archivo PNG
saveas(gcf, 'AV_Improvement_Percentage.png');
