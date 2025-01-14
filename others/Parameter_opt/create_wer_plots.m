function create_wer_plots(file_name, plot_title, plot_filename, beam_size, lenpen, wer)
    % Crear una cuadrícula para los valores de Beam Size y Lenpen
    [beam_grid, lenpen_grid] = meshgrid(unique(beam_size), unique(lenpen));

    % Interpolar los valores de WER para la cuadrícula
    wer_grid = griddata(beam_size, lenpen, wer, beam_grid, lenpen_grid, 'linear');

    % Encontrar el mínimo de WER y sus coordenadas
    [min_wer, min_idx] = min(wer);
    min_beam_size = beam_size(min_idx);
    min_lenpen = lenpen(min_idx);

    % Crear la gráfica de superficie (vista 3D)
    figure;
    surf(beam_grid, lenpen_grid, wer_grid);

    % Personalizar la gráfica
    xlabel('Beam Size', 'FontSize', 14); % Etiqueta para el eje X
    ylabel('Length penalty', 'FontSize', 14);    % Etiqueta para el eje Y
    zlabel('WER', 'FontSize', 14);       % Etiqueta para el eje Z
    title(plot_title, 'FontSize', 16);   % Título

    % Opciones estéticas
    colorbar('FontSize', 18);            % Añadir barra de colores para interpretar WER
    colormap('jet');                     % Aplicar el colormap tipo jet
    shading interp;                      % Suavizar la superficie
    view(45, 30);                        % Ajustar ángulo de vista: 45° horizontal y 30° vertical
    set(gca, 'FontSize', 18);            % Ajustar tamaño de los números de los ejes

    % Añadir leyenda para el mínimo
    legend_text = sprintf('\\bfWER_{min}:\\rm %.2f%% \n\\it(Beam Size: %d, Lenpen: %.1f)\\rm', ...
                          min_wer, min_beam_size, min_lenpen);
    annotation('textbox', [0.66, 0.8, 0.1, 0.1], 'String', legend_text, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 16, ...
               'Interpreter', 'tex');

    % Ajustar el tamaño de la figura para mayor resolución
    set(gcf, 'Position', [100, 100, 1200, 800]); % Tamaño de la figura en píxeles (ajustable)

    % Guardar la gráfica 3D con alta resolución
    print(plot_filename, '-dpng', '-r300'); % Guardar en PNG con resolución de 300 dpi

    % Crear la gráfica desde arriba (vista 2D)
    figure;

    % Graficar solo los valores con datos
    surf(beam_grid, lenpen_grid, wer_grid);

    % Personalizar la gráfica
    xlabel('Beam Size', 'FontSize', 20); % Etiqueta para el eje X
    ylabel('Length penalty', 'FontSize', 20);    % Etiqueta para el eje Y
    title(['Top View of ' plot_title], 'FontSize', 20); % Título

    % Ajustar límites de los ejes según los valores disponibles
    xlim([min(beam_size) max(beam_size)]); % Ajustar límites del eje X
    ylim([min(lenpen) max(lenpen)]);       % Ajustar límites del eje Y

    % Opciones estéticas
    colorbar('FontSize', 18);            % Añadir barra de colores para interpretar WER
    colormap('jet');                     % Aplicar el colormap tipo jet
    shading interp;                      % Suavizar la superficie
    view(0, 90);                         % Ajustar ángulo de vista para que sea desde arriba
    set(gca, 'FontSize', 18);            % Ajustar tamaño de los números de los ejes

    % Añadir leyenda para el mínimo
    annotation('textbox', [0.67, 0.8, 0.1, 0.1], 'String', legend_text, ...
               'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 16, ...
               'Interpreter', 'tex');

    % Ajustar el tamaño de la figura para mayor resolución en 2D
    set(gcf, 'Position', [100, 100, 1200, 800]); % Tamaño de la figura en píxeles (ajustable)

    % Guardar la gráfica 2D con alta resolución
    print(['top_view_' plot_filename], '-dpng', '-r300'); % Guardar en PNG con resolución de 300 dpi
end
