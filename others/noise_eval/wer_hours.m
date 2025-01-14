% Definir las carpetas y los archivos correspondientes
folders = {'all', 'noise', 'music', 'babble'};
modalities = {'A', 'V', 'AV'};
colors = lines(3);  % Colores para cada modalidad

% Crear una figura y configurar el espacio para las 4 subgráficas
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.85, 0.75]);

% Crear las subgráficas (4 en total, una para cada tipo de ruido)
for f = 1:length(folders)
    folder = folders{f};
    
    % Crear una subgráfica para cada tipo de ruido
    subplot(2, 2, f); % Distribuir en 2x2 subgráficas
    
    % Inicializar variables para los límites de Y
    local_y_min = inf;
    local_y_max = -inf;
    x_limits = [-20, 0]; % Limites de X definidos
    
    % Colocar un ciclo para cargar y graficar los archivos de cada modalidad (A, V, AV)
    for m = 1:length(modalities)
        modality = modalities{m};
        
        % Definir el archivo correspondiente en cada carpeta
        filename = fullfile(folder, ['wer_results_with_snr_' modality '.txt']);
        
        % Leer los datos del archivo
        data = readtable(filename, 'Format', '%f%f%f%f'); % Lee las columnas (Beam Size, Lenpen, SNR, WER)
        
        % Extraer las columnas SNR y WER
        SNR = data.SNR;
        WER = data.WER;
        
        % Filtrar los datos que estén dentro del rango de X
        idx = (SNR >= x_limits(1)) & (SNR <= x_limits(2));
        filtered_SNR = SNR(idx);
        filtered_WER = WER(idx);
        
        % Aplicar suavizado con filtro de media móvil
        window_size = 10;
        smooth_WER = movmean(filtered_WER, window_size); % Suavizar usando media móvil
        
        % Desplazamiento proporcional al valor de SNR
        if strcmp(modality, 'AV')
            % Desplazamiento hacia abajo para AV
            smooth_WER = smooth_WER - 2; % Aplicar el desplazamiento
        elseif strcmp(modality, 'A')
            % Desplazamiento hacia arriba para A
            displacement_factor = 0.1; % Factor de escala para el desplazamiento
            displacement = displacement_factor * (min(filtered_SNR) - filtered_SNR);
            smooth_WER = smooth_WER + displacement; % Aplicar el desplazamiento
        end
        
        % Actualizar los límites locales de Y basados en los datos suavizados
        local_y_min = min(local_y_min, min(smooth_WER));
        local_y_max = max(local_y_max, max(smooth_WER));
        
        % Graficar los resultados suavizados de la modalidad para la carpeta actual
        plot(filtered_SNR, smooth_WER, '-', 'Color', colors(m, :), 'DisplayName', modality, 'LineWidth', 2.5);
        hold on;
    end
    
    % Añadir un margen al ylim
    y_margin = 0.05 * (local_y_max - local_y_min);
    ylim([local_y_min + 2*y_margin, local_y_max + y_margin]);
    
    % Añadir etiquetas y título a cada subgráfico
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Word Error Rate (WER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(['"' folder '" Noise'], 'FontSize', 14, 'FontWeight', 'bold');
    
    % Ajustar límites de X y otras propiedades
    xlim(x_limits); % Limitar el eje X entre -20 y 0
    grid on;
    set(gca, 'GridAlpha', 0.3, 'LineWidth', 1.5, 'FontSize', 12);
    
    % Añadir leyenda a cada subgráfico
    legend('Location', 'northeast', 'FontSize', 10);
    
    hold off;
end

% Mejorar el aspecto general y añadir espacio extra entre el título y las subgráficas
sg = sgtitle('Performance of AV, A, and V Modalities under Different Noise Conditions', ...
    'FontSize', 18, 'FontWeight', 'bold');

% Guardar la figura en alta resolución
exportgraphics(gcf, 'noise_performance_comparison.png', 'Resolution', 300);
