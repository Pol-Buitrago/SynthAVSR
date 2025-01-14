% Código fuera de la función, en el script principal

% Archivos de entrada y nombres de salida
files = {'wer_results_AV.txt', 'wer_results_A.txt', 'wer_results_V.txt'};
titles = {'WER for AV Data', 'WER for A Data', 'WER for V Data'};
file_prefixes = {'AV', 'A', 'V'};

% Crear las gráficas para cada archivo
for i = 1:length(files)
    data = readtable(files{i}, 'Delimiter', ',', 'ReadVariableNames', true);
    create_wer_plots(files{i}, titles{i}, ['wer_surface_plot_' file_prefixes{i} '.png'], ...
                     data.BeamSize, data.Lenpen, data.WER);
end

% Calcular la media de WER de los tres archivos
%data_AV = readtable('wer_results_AV.txt', 'Delimiter', ',', 'ReadVariableNames', true);
%data_A = readtable('wer_results_A.txt', 'Delimiter', ',', 'ReadVariableNames', true);
%data_V = readtable('wer_results_V.txt', 'Delimiter', ',', 'ReadVariableNames', true);

% Promediar los WER
%averaged_wer = (data_AV.WER + data_A.WER + data_V.WER) / 3;

% Llamar a la función para crear las gráficas de la media
%create_wer_plots('wer_average.txt', 'Average WER for All Datasets', 'wer_surface_plot_average.png', ...
%                 data_AV.BeamSize, data_AV.Lenpen, averaged_wer);
