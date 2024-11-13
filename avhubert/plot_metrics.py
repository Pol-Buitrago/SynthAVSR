import re
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ruta del archivo log
log_file_path = '/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/finetune/20241113_011531-AV_1GPU/hydra_train.log'

# Extraemos el número del archivo log (el número después de "finetune_avhubert_")
log_number = re.search(r'finetune_avhubert_(\d+)\.log', log_file_path)
if log_number:
    log_number = log_number.group(1)
else:
    log_number = "unknown"

# Definimos la ruta de la carpeta para guardar las gráficas
plots_folder = 'plots/finetune_' + log_number

# Creamos la carpeta si no existe
os.makedirs(plots_folder, exist_ok=True)

# Inicializamos las listas para guardar los valores
epochs_train = []
train_losses = []

epochs_valid = []
valid_losses = []

epochs_train_accuracy = []
train_accuracies = []

epochs_valid_accuracy = []
valid_accuracies = []

epochs_train_lr = []
train_lrs = []

epochs_train_wps = []
train_wpss = []

epochs_train_gnorm = []
train_gnorms = []

epochs_train_bsz = []
train_bszs = []

epochs_train_num_updates = []
train_num_updates = []

# Expresión regular para buscar las líneas que contienen las variables
train_loss_pattern = r'"epoch":\s*(\d+),\s*"train_loss":\s*"([\d.]+)"'
valid_loss_pattern = r'"epoch":\s*(\d+),\s*"valid_loss":\s*"([\d.]+)"'
train_accuracy_pattern = r'"epoch":\s*(\d+)[^}]*"train_accuracy":\s*"([\d.]+)"'
valid_accuracy_pattern = r'"epoch":\s*(\d+)[^}]*"valid_accuracy":\s*"([\d.]+)"'

train_lr_pattern = r'"epoch":\s*(\d+)[^}]*"train_lr":\s*"([\d.e+-]+)"'
train_wps_pattern = r'"epoch":\s*(\d+)[^}]*"train_wps":\s*"([\d.]+)"'
train_gnorm_pattern = r'"epoch":\s*(\d+)[^}]*"train_gnorm":\s*"([\d.]+)"'
train_bsz_pattern = r'"epoch":\s*(\d+)[^}]*"train_bsz":\s*"([\d.]+)"'
train_num_updates_pattern = r'"epoch":\s*(\d+)[^}]*"train_num_updates":\s*"(\d+)"'


# Leemos el archivo log y buscamos las líneas que contienen "train_loss", "valid_loss", "train_accuracy" y "valid_accuracy"
with open(log_file_path, 'r') as f:
    for line in f:
        # Buscar train_loss
        match_train_loss = re.search(train_loss_pattern, line)
        if match_train_loss:
            epoch_train = int(match_train_loss.group(1))
            train_loss = float(match_train_loss.group(2))
            epochs_train.append(epoch_train)
            train_losses.append(train_loss)

        # Buscar valid_loss
        match_valid_loss = re.search(valid_loss_pattern, line)
        if match_valid_loss:
            epoch_valid = int(match_valid_loss.group(1))
            valid_loss = float(match_valid_loss.group(2))
            epochs_valid.append(epoch_valid)
            valid_losses.append(valid_loss)

        # Buscar train_accuracy
        match_train_accuracy = re.search(train_accuracy_pattern, line)
        if match_train_accuracy:
            epoch_train_accuracy = int(match_train_accuracy.group(1))
            train_accuracy = float(match_train_accuracy.group(2))
            epochs_train_accuracy.append(epoch_train_accuracy)
            train_accuracies.append(train_accuracy)

        # Buscar valid_accuracy
        match_valid_accuracy = re.search(valid_accuracy_pattern, line)
        if match_valid_accuracy:
            epoch_valid_accuracy = int(match_valid_accuracy.group(1))
            valid_accuracy = float(match_valid_accuracy.group(2))
            epochs_valid_accuracy.append(epoch_valid_accuracy)
            valid_accuracies.append(valid_accuracy)
        
        # Buscar train_lr
        match_train_lr = re.search(train_lr_pattern, line)
        if match_train_lr:
            epoch_train_lr = int(match_train_lr.group(1))
            train_lr = float(match_train_lr.group(2))
            epochs_train_lr.append(epoch_train_lr)
            train_lrs.append(train_lr)

        # Buscar train_wps
        match_train_wps = re.search(train_wps_pattern, line)
        if match_train_wps:
            epoch_train_wps = int(match_train_wps.group(1))
            train_wps = float(match_train_wps.group(2))
            epochs_train_wps.append(epoch_train_wps)
            train_wpss.append(train_wps)

        # Buscar train_gnorm
        match_train_gnorm = re.search(train_gnorm_pattern, line)
        if match_train_gnorm:
            epoch_train_gnorm = int(match_train_gnorm.group(1))
            train_gnorm = float(match_train_gnorm.group(2))
            epochs_train_gnorm.append(epoch_train_gnorm)
            train_gnorms.append(train_gnorm)

        # Buscar train_bsz
        match_train_bsz = re.search(train_bsz_pattern, line)
        if match_train_bsz:
            epoch_train_bsz = int(match_train_bsz.group(1))
            train_bsz = float(match_train_bsz.group(2))
            epochs_train_bsz.append(epoch_train_bsz)
            train_bszs.append(train_bsz)

        # Buscar train_num_updates
        match_train_num_updates = re.search(train_num_updates_pattern, line)
        if match_train_num_updates:
            epoch_train_num_updates = int(match_train_num_updates.group(1))
            train_num_update = int(match_train_num_updates.group(2))
            epochs_train_num_updates.append(epoch_train_num_updates)
            train_num_updates.append(train_num_update)

# Verificamos que se hayan encontrado datos
if epochs_train and train_losses:
    # Convertimos las listas a arrays de numpy para facilidad en operaciones
    epochs_train = np.array(epochs_train)
    train_losses = np.array(train_losses)

    # Aplicamos un suavizado utilizando un filtro de media móvil (window size = 5)
    window_size = 5
    smoothed_train_losses = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')

    # Encontramos el mínimo de train_loss y su índice
    min_train_loss_idx = np.argmin(smoothed_train_losses)
    min_train_loss = smoothed_train_losses[min_train_loss_idx]
    min_train_epoch = epochs_train[min_train_loss_idx]

    # Creamos la gráfica para train_loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train[:len(smoothed_train_losses)], smoothed_train_losses, linestyle='-', color='b', label='Smoothed Train Loss')
    plt.title(f'Train Loss vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.grid(True)
    plt.legend()

    # Marcamos el mínimo de train_loss
    plt.scatter(min_train_epoch, min_train_loss, color='blue', zorder=5)
    plt.annotate(f'{min_train_loss:.4f}', 
                 (min_train_epoch, min_train_loss), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='blue')

    # Guardamos la gráfica de train_loss
    train_loss_plot_path = os.path.join(plots_folder, f'train_loss_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(train_loss_plot_path)
    print(f"Gráfica de train_loss guardada como '{train_loss_plot_path}'")

if epochs_valid and valid_losses:
    # Convertimos las listas a arrays de numpy para facilidad en operaciones
    epochs_valid = np.array(epochs_valid)
    valid_losses = np.array(valid_losses)

    # Aplicamos un suavizado utilizando un filtro de media móvil (window size = 5)
    smoothed_valid_losses = np.convolve(valid_losses, np.ones(window_size)/window_size, mode='valid')

    # Encontramos el mínimo de valid_loss y su índice
    min_valid_loss_idx = np.argmin(smoothed_valid_losses)
    min_valid_loss = smoothed_valid_losses[min_valid_loss_idx]
    min_valid_epoch = epochs_valid[min_valid_loss_idx]

    # Creamos la gráfica para valid_loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_valid[:len(smoothed_valid_losses)], smoothed_valid_losses, linestyle='-', color='r', label='Smoothed Valid Loss')
    plt.title(f'Valid Loss vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Valid Loss')
    plt.grid(True)
    plt.legend()

    # Marcamos el mínimo de valid_loss
    plt.scatter(min_valid_epoch, min_valid_loss, color='red', zorder=5)
    plt.annotate(f'{min_valid_loss:.4f}', 
                 (min_valid_epoch, min_valid_loss), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='red')

    # Guardamos la gráfica de valid_loss
    valid_loss_plot_path = os.path.join(plots_folder, f'valid_loss_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(valid_loss_plot_path)
    print(f"Gráfica de valid_loss guardada como '{valid_loss_plot_path}'")

    # Creamos la tercera gráfica, con ambas pérdidas en el mismo gráfico
    plt.figure(figsize=(10, 6))

    # Grafica de train_loss
    plt.plot(epochs_train[:len(smoothed_train_losses)], smoothed_train_losses, linestyle='-', color='b', label='Train Loss')

    # Grafica de valid_loss
    plt.plot(epochs_valid[:len(smoothed_valid_losses)], smoothed_valid_losses, linestyle='-', color='r', label='Valid Loss')

    # Marcamos los mínimos de ambas curvas
    plt.scatter(min_train_epoch, min_train_loss, color='blue', zorder=5)
    plt.annotate(f'{min_train_loss:.4f}', 
                 (min_train_epoch, min_train_loss), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='blue')

    plt.scatter(min_valid_epoch, min_valid_loss, color='red', zorder=5)
    plt.annotate(f'{min_valid_loss:.4f}', 
                 (min_valid_epoch, min_valid_loss), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='red')

    plt.title(f'Train Loss and Valid Loss vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Guardamos la gráfica combinada
    combined_loss_plot_path = os.path.join(plots_folder, f'combined_loss_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(combined_loss_plot_path)
    print(f"Gráfica combinada de loss guardada como '{combined_loss_plot_path}'")
    
# Graficamos train_accuracy
if epochs_train_accuracy and train_accuracies:
    # Convertimos las listas a arrays de numpy
    epochs_train_accuracy = np.array(epochs_train_accuracy)
    train_accuracies = np.array(train_accuracies)

    # Suavizado de train_accuracy
    smoothed_train_accuracies = np.convolve(train_accuracies, np.ones(window_size)/window_size, mode='valid')

    # Encontramos el máximo de train_accuracy y su índice
    max_train_accuracy_idx = np.argmax(smoothed_train_accuracies)
    max_train_accuracy = smoothed_train_accuracies[max_train_accuracy_idx]
    max_train_epoch = epochs_train_accuracy[max_train_accuracy_idx]

    # Crear la gráfica de train_accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_train_accuracy[:len(smoothed_train_accuracies)], smoothed_train_accuracies, linestyle='-', color='b', label='Smoothed Train Accuracy')
    plt.title(f'Train Accuracy vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.grid(True)
    plt.legend()

    # Marcamos el máximo de train_accuracy
    plt.scatter(max_train_epoch, max_train_accuracy, color='blue', zorder=5)
    plt.annotate(f'{max_train_accuracy:.4f}', 
                 (max_train_epoch, max_train_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='blue')

    # Guardamos la gráfica de train_accuracy
    train_accuracy_plot_path = os.path.join(plots_folder, f'train_accuracy_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(train_accuracy_plot_path)
    print(f"Gráfica de train_accuracy guardada como '{train_accuracy_plot_path}'")

# Graficamos valid_accuracy
if epochs_valid_accuracy and valid_accuracies:
    # Convertimos las listas a arrays de numpy
    epochs_valid_accuracy = np.array(epochs_valid_accuracy)
    valid_accuracies = np.array(valid_accuracies)

    # Suavizado de valid_accuracy
    smoothed_valid_accuracies = np.convolve(valid_accuracies, np.ones(window_size)/window_size, mode='valid')

    # Encontramos el máximo de valid_accuracy y su índice
    max_valid_accuracy_idx = np.argmax(smoothed_valid_accuracies)
    max_valid_accuracy = smoothed_valid_accuracies[max_valid_accuracy_idx]
    max_valid_epoch = epochs_valid_accuracy[max_valid_accuracy_idx]

    # Crear la gráfica de valid_accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_valid_accuracy[:len(smoothed_valid_accuracies)], smoothed_valid_accuracies, linestyle='-', color='r', label='Smoothed Valid Accuracy')
    plt.title(f'Valid Accuracy vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Valid Accuracy')
    plt.grid(True)
    plt.legend()

    # Marcamos el máximo de valid_accuracy
    plt.scatter(max_valid_epoch, max_valid_accuracy, color='red', zorder=5)
    plt.annotate(f'{max_valid_accuracy:.4f}', 
                 (max_valid_epoch, max_valid_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='red')

    # Guardamos la gráfica de valid_accuracy
    valid_accuracy_plot_path = os.path.join(plots_folder, f'valid_accuracy_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(valid_accuracy_plot_path)
    print(f"Gráfica de valid_accuracy guardada como '{valid_accuracy_plot_path}'")

    # Graficamos las dos métricas de accuracy en una gráfica combinada
    plt.figure(figsize=(10, 6))

    # Gráfica de train_accuracy
    plt.plot(epochs_train_accuracy[:len(smoothed_train_accuracies)], smoothed_train_accuracies, linestyle='-', color='b', label='Train Accuracy')

    # Gráfica de valid_accuracy
    plt.plot(epochs_valid_accuracy[:len(smoothed_valid_accuracies)], smoothed_valid_accuracies, linestyle='-', color='r', label='Valid Accuracy')

    # Marcamos los máximos de ambas curvas
    plt.scatter(max_train_epoch, max_train_accuracy, color='blue', zorder=5)
    plt.annotate(f'{max_train_accuracy:.4f}', 
                 (max_train_epoch, max_train_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='blue')

    plt.scatter(max_valid_epoch, max_valid_accuracy, color='red', zorder=5)
    plt.annotate(f'{max_valid_accuracy:.4f}', 
                 (max_valid_epoch, max_valid_accuracy), 
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 color='red')

    plt.title(f'Train Accuracy and Valid Accuracy vs Epoch (Smoothed) - {log_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Guardamos la gráfica combinada de accuracy
    combined_accuracy_plot_path = os.path.join(plots_folder, f'combined_accuracy_vs_epoch_smoothed_{log_number}.png')
    plt.savefig(combined_accuracy_plot_path)
    print(f"Gráfica combinada de accuracy guardada como '{combined_accuracy_plot_path}'")

# Generación de gráficas para las nuevas variables
def plot_and_save(x, y, ylabel, color, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color=color, linestyle='-', label=title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{title} vs Epoch - {log_number}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_folder, filename))
    print(f"Gráfica de {title} guardada como '{filename}'")

# Graficar y guardar cada nueva variable
if epochs_train_lr:
    plot_and_save(epochs_train_lr, train_lrs, 'Learning Rate', 'purple', 'Learning Rate', f'train_lr_vs_epoch_{log_number}.png')
if epochs_train_wps:
    plot_and_save(epochs_train_wps, train_wpss, 'Words per Second', 'orange', 'Train WPS', f'train_wps_vs_epoch_{log_number}.png')
if epochs_train_gnorm:
    plot_and_save(epochs_train_gnorm, train_gnorms, 'Gradient Norm', 'green', 'Train Gradient Norm', f'train_gnorm_vs_epoch_{log_number}.png')
if epochs_train_bsz:
    plot_and_save(epochs_train_bsz, train_bszs, 'Batch Size', 'blue', 'Train Batch Size', f'train_bsz_vs_epoch_{log_number}.png')
if epochs_train_num_updates:
    plot_and_save(epochs_train_num_updates, train_num_updates, 'Num Updates', 'red', 'Train Num Updates', f'train_num_updates_vs_epoch_{log_number}.png')

# Define la carpeta de salida
output_folder = plots_folder + "/combined_plots"
os.makedirs(output_folder, exist_ok=True)

# Función para cargar y organizar imágenes según el prefijo
def combine_images_by_prefix(prefix):
    # Encuentra todas las imágenes con el prefijo en la carpeta de plots
    images = [file for file in os.listdir(plots_folder) if file.startswith(prefix) and file.endswith('.png')]
    
    if not images:
        print(f"No se encontraron imágenes con el prefijo '{prefix}' en {plots_folder}.")
        return
    
    # Carga todas las imágenes
    img_list = [Image.open(os.path.join(plots_folder, img)) for img in images]
    
    # Calcular el tamaño de cada imagen y la cantidad de filas y columnas
    img_width, img_height = img_list[0].size
    columns = 2
    rows = (len(img_list) + columns - 1) // columns  # Redondear hacia arriba
    
    # Crear una nueva imagen para combinar las gráficas
    combined_width = columns * img_width
    combined_height = rows * img_height
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')
    
    # Pegar las imágenes en la nueva imagen, organizadas en una matriz 2xN
    for i, img in enumerate(img_list):
        row = i // columns  # Determinar en qué fila va la imagen
        col = i % columns   # Determinar en qué columna va la imagen
        x_offset = col * img_width
        y_offset = row * img_height
        combined_image.paste(img, (x_offset, y_offset))
    
    # Guarda la imagen combinada en la carpeta de salida
    combined_image.save(os.path.join(output_folder, f"{prefix}combined.png"))
    print(f"Imagen combinada para '{prefix}' guardada en '{output_folder}'.")

# Genera las imágenes combinadas para cada prefijo de interés
combine_images_by_prefix("train_")
combine_images_by_prefix("valid_")
combine_images_by_prefix("combined_")

print(f"Todas las imágenes combinadas han sido guardadas en la carpeta '{output_folder}'.")
