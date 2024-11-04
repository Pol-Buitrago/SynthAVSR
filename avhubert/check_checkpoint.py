import torch

# Ruta al archivo del checkpoint
checkpoint_path = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/AVSR_Finetuned_Models/Spanish_ES/best_ckpt.pt"  # Cambia esto a la ruta de tu archivo .pt

# Cargar el checkpoint
checkpoint = torch.load(checkpoint_path)

# Imprimir información general del checkpoint
print("Contenido del checkpoint:")
for key in checkpoint.keys():
    print(key)

# Si el modelo tiene un estado guardado, imprimir las dimensiones
if 'model' in checkpoint:
    model_state_dict = checkpoint['model']
    print("\nDimensiones de los parámetros del modelo:")
    for param_name, param_value in model_state_dict.items():
        print(f"{param_name}: {param_value.shape}")
