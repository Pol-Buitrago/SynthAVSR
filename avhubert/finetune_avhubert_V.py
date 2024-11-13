import os
import subprocess
from datetime import datetime


# Ruta base para simplificar las rutas repetidas
BASE_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert"

def finetune_avhubert(
    config_dir=f"{BASE_DIR}/conf/finetune",
    config_name="base_vox_433h_V.yaml",
    data_dir=f"{BASE_DIR}/data/datasets/dataset_ES/LIP_RTVE/model_data",
    label_dir=f"{BASE_DIR}/data/datasets/dataset_ES/LIP_RTVE/model_data",
    tokenizer_model=f"{BASE_DIR}/data/datasets/dataset_ES/LIP_RTVE/spm1000/spm_unigram1000.model",
    checkpoint_path=f"{BASE_DIR}/checkpoints/Pretrained_Models/large_vox_iter5.pt",
):
    """
    Función para ejecutar el fine-tuning de un modelo AV-HuBERT.
    """
    # Generar un identificador único para cada ejecución basado en la fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{BASE_DIR}/experiment/finetune/{timestamp}"

    # Crear el directorio de ejecución si no existe
    os.makedirs(run_dir, exist_ok=True)
    
    # Comando de entrenamiento
    cmd = (
        f"PYTHONPATH=/gpfs/projects/bsc88/speech/research/repos/av_hubert/fairseq "
        f"fairseq-hydra-train --config-dir {config_dir} --config-name {config_name} "
        f"task.data={data_dir} "
        f"task.label_dir={label_dir} "
        f"task.tokenizer_bpe_model={tokenizer_model} "
        f"model.w2v_path={checkpoint_path} "
        f"hydra.run.dir={run_dir} "
        f"common.user_dir=`pwd`"
    )

    
    # Imprimir el comando para verificación
    print("Ejecutando el fine-tuning con el siguiente comando:")
    print(cmd)
    
    # Ejecutar el comando con subprocess para manejar excepciones
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    # Ejecutar el fine-tuning con los valores por defecto definidos en la función
    finetune_avhubert()
