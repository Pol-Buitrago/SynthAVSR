import os
import sys

def finetune_avhubert(config_dir, config_name, data_dir, label_dir, tokenizer_model, checkpoint_path, run_dir):
    """
    Función para ejecutar el fine-tuning de un modelo AV-HuBERT.

    :param config_dir: Directorio donde se encuentra el archivo de configuración.
    :param config_name: Nombre del archivo de configuración.
    :param data_dir: Ruta al directorio con los datos (.tsv).
    :param label_dir: Ruta al directorio con las etiquetas (.wrd).
    :param tokenizer_model: Ruta al modelo BPE para tokenización.
    :param checkpoint_path: Ruta al modelo preentrenado.
    :param run_dir: Directorio donde se guardarán los resultados del entrenamiento.
    """
    
    # Comando de entrenamiento
    cmd = (
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
    
    # Ejecutar el comando
    os.system(cmd)

if __name__ == "__main__":
    # Valores por defecto
    DEFAULT_CONFIG_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/conf/finetune"
    DEFAULT_CONFIG_NAME = "self_large_vox_433h.yaml"
    DEFAULT_DATA_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE"
    DEFAULT_LABEL_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/data/datasets/dataset_ES/LIP_RTVE/model_data"
    DEFAULT_TOKENIZER_MODEL = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/tokenizer.model"
    DEFAULT_CHECKPOINT_PATH = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/checkpoints/Pretrained_Models/large_vox_iter5.pt"
    DEFAULT_RUN_DIR = "/gpfs/projects/bsc88/speech/research/repos/av_hubert/avhubert/experiment/finetune"

    # Obtener los parámetros desde la línea de comandos (si existen)
    config_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_DIR
    config_name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG_NAME
    data_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DATA_DIR
    label_dir = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_LABEL_DIR
    tokenizer_model = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_TOKENIZER_MODEL
    checkpoint_path = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_CHECKPOINT_PATH
    run_dir = sys.argv[7] if len(sys.argv) > 7 else DEFAULT_RUN_DIR

    # Ejecutar el fine-tuning
    finetune_avhubert(config_dir, config_name, data_dir, label_dir, tokenizer_model, checkpoint_path, run_dir)
